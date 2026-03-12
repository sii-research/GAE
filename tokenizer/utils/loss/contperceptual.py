import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from tokenizer.utils.loss.gan.model import NLayerDiscriminator
from tokenizer.utils.loss.lpips.lpips import LPIPS

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight
        
class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge",
                 
                 semantic_loss_weight=1.0,      
                 use_vf_loss=False,              
                 vf_cos_margin=0.5,
                 vf_distmat_margin=0.25,
                 vf_cos_weight=1.0,
                 vf_distmat_weight=1.0,
                 use_adaptive_sp_weight=False   
                ):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight 
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        

        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        print(f"self.disc_factor = {disc_factor}")
        print(f"self.discriminator_weight = {disc_weight}")
        print(f"self.disc_loss = {self.disc_loss}")
        self.disc_conditional = disc_conditional
        

        self.semantic_loss_weight = semantic_loss_weight
        self.use_vf_loss = use_vf_loss
        self.vf_cos_margin = vf_cos_margin
        self.vf_distmat_margin = vf_distmat_margin
        self.vf_cos_weight = vf_cos_weight
        self.vf_distmat_weight = vf_distmat_weight
        self.use_adaptive_sp_weight = use_adaptive_sp_weight
        
    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is None:
            return torch.tensor(0.0, device=nll_loss.device)

        with torch.cuda.amp.autocast(enabled=False):
            nll_loss_fp32 = nll_loss.float()
            g_loss_fp32 = g_loss.float()

            nll_grads = torch.autograd.grad(nll_loss_fp32, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss_fp32, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def calculate_adaptive_sp_weight(self, nll_loss, sp_loss, enc_last_layer):
        if enc_last_layer is None:
             return torch.tensor(0.0, device=nll_loss.device)

        with torch.cuda.amp.autocast(enabled=False):
            nll_loss_fp32 = nll_loss.float()
            sp_loss_fp32 = sp_loss.float()

            try:
                nll_grads = torch.autograd.grad(
                    outputs=nll_loss_fp32, 
                    inputs=enc_last_layer, 
                    retain_graph=True  
                )[0]

                sp_grads = torch.autograd.grad(
                    outputs=sp_loss_fp32, 
                    inputs=enc_last_layer,
                    retain_graph=True  
                )[0]

            except RuntimeError as e:
                return torch.tensor(0.0, device=nll_loss.device)

        sp_weight = torch.norm(nll_grads) / (torch.norm(sp_grads) + 1e-4)
        sp_weight = torch.clamp(sp_weight, 0.0, 1e8).detach()
        sp_weight = sp_weight * self.semantic_loss_weight
        return sp_weight

    def calculate_vf_loss(self, h, h_frozen):
        N_b, N_c, N_h, N_w = h.shape
        h_frozen_d = h_frozen.detach()

        cos_sim = F.cosine_similarity(h, h_frozen_d, dim=1)
        vf_loss_cos = F.relu(1.0 - self.vf_cos_margin - cos_sim).mean()

        h_flat = h.view(N_b, N_c, -1)
        h_frozen_flat = h_frozen_d.view(N_b, N_c, -1)
        h_norm = F.normalize(h_flat, p=2, dim=1)
        h_frozen_norm = F.normalize(h_frozen_flat, p=2, dim=1)
        h_sim_mat = torch.einsum('bci,bcj->bij', h_norm, h_norm)
        h_frozen_sim_mat = torch.einsum('bci,bcj->bij', h_frozen_norm, h_frozen_norm)
        diff = torch.abs(h_sim_mat - h_frozen_sim_mat)
        vf_loss_distmat = F.relu(diff - self.vf_distmat_margin).mean()

        total_vf_loss = (vf_loss_cos * self.vf_cos_weight + 
                         vf_loss_distmat * self.vf_distmat_weight)
        
        return total_vf_loss, vf_loss_cos.detach(), vf_loss_distmat.detach()

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None,
                h_semantic=None,
                h_frozen_semantic=None,
                enc_last_layer=None
               ):
        
        l1_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.pixel_weight > 0:
            l1_loss = l1_loss * self.pixel_weight
        
        p_loss = torch.tensor(0.0, device=inputs.device)
        
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = l1_loss + self.perceptual_weight * p_loss
        else:
            rec_loss = l1_loss 

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        
        weighted_nll_loss = torch.mean(weighted_nll_loss)
        nll_loss = torch.mean(nll_loss)

        if posteriors is not None:
            kl_loss = posteriors.kl() 
            kl_loss = torch.mean(kl_loss)
        else:
            kl_loss = torch.tensor(0.0, device=inputs.device)
            
        if optimizer_idx == 0:
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = torch.tensor(self.discriminator_weight, device=inputs.device)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            sp_loss = torch.tensor(0.0, device=inputs.device)
            sp_loss_cos = torch.tensor(0.0, device=inputs.device)
            sp_loss_distmat = torch.tensor(0.0, device=inputs.device)
            
            if h_semantic is not None and h_frozen_semantic is not None:
                if self.use_vf_loss:
                    sp_loss, sp_loss_cos, sp_loss_distmat = self.calculate_vf_loss(h_semantic, h_frozen_semantic)
                else:
                    sp_loss = F.mse_loss(h_semantic, h_frozen_semantic.detach())
            
            if self.use_adaptive_sp_weight and sp_loss > 0 and enc_last_layer is not None:
                try:
                    sp_weight = self.calculate_adaptive_sp_weight(nll_loss, sp_loss, enc_last_layer=enc_last_layer)
                except RuntimeError:
                    assert not self.training
                    sp_weight = torch.tensor(0.0)
            else:
                sp_weight = self.semantic_loss_weight
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            
        
            loss = (weighted_nll_loss + 
                    self.kl_weight * kl_loss + 
                    d_weight * disc_factor * g_loss + 
                    sp_weight * sp_loss)


            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(), 
                   "{}/l1_loss".format(split): l1_loss.detach().mean(),
                   "{}/p_loss".format(split): p_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   "{}/sp_loss".format(split): sp_loss.detach().mean(),
                   "{}/sp_weight".format(split): sp_weight.detach() if isinstance(sp_weight, torch.Tensor) else torch.tensor(sp_weight),
                   "{}/sp_loss_cos".format(split): sp_loss_cos.mean(),
                   "{}/sp_loss_distmat".format(split): sp_loss_distmat.mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log