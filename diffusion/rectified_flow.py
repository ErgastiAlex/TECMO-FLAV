import torch
from tqdm import tqdm
import numpy as np
import math
class RectifiedFlow():
    def __init__(self, num_timesteps, warmup_timesteps = 10, noise_scale=1.0, init_type='gaussian', eps=1, sampling='logit', window_size=8):
        """
        eps: A `float` number. The smallest time step to sample from.
        """
        self.num_timesteps = num_timesteps

        self.warmup_timesteps = warmup_timesteps*num_timesteps
        self.T = 1000.
        self.noise_scale = noise_scale
        self.init_type = init_type
        self.eps = eps

        self.window_size = window_size

        self.sampling = sampling

    def logit(self, x):
        return torch.log(x / (1 - x))

    def logit_normal(self, x, mu=0, sigma=1):
        return 1 / (sigma * math.sqrt(2 * torch.pi) * x * (1 - x)) * torch.exp(-(self.logit(x) - mu) ** 2 / (2 * sigma ** 2))

    def training_loss(self, model, v, a, model_kwargs):
        """
        v: [B, T, C, H, W]
        a: [B, T, N, F]
        """

        B,T = v.shape[:2]
            
        tw = torch.rand((v.shape[0],1), device=v.device)

        window_indexes = torch.linspace(0, self.window_size-1, steps=self.window_size, device=v.device).unsqueeze(0).repeat(B,1)
        
        rollout = torch.bernoulli(torch.tensor(0.8).repeat(B).to(v.device)).bool()
        t_rollout = (window_indexes+tw)/self.window_size
        
        t_pre_rollout = window_indexes/self.window_size + tw

        t = torch.where(rollout.unsqueeze(1).repeat(1,self.window_size), t_rollout, t_pre_rollout)
        t = 1 - t # swap 0 and 1, since 1 is full image and 0 is full noise

        t = torch.clamp(t, 0+1e-6, 1-1e-6)

        if self.sampling == 'logit':
            weigths = self.logit_normal(t, mu=0, sigma=1)
        else:
            weigths = torch.ones_like(t)

        B, T = t.shape

        v_z0 = self.get_z0(v).to(v.device)
        a_z0 = self.get_z0(a).to(a.device)

        t_video = t.view(B,T,1,1,1).repeat(1,1,v.shape[2], v.shape[3], v.shape[4]) 
        t_audio = t.view(B,T,1,1,1).repeat(1,1,a.shape[2], a.shape[3], a.shape[4])

        perturbed_video = t_video*v + (1-t_video)*v_z0
        perturbed_audio = t_audio*a + (1-t_audio)*a_z0

        t_rf = t*(self.T-self.eps) + self.eps
        score_v, score_a = model(perturbed_video, perturbed_audio, t_rf, **model_kwargs)

        # score_v = [B, T, C, H, W]
        # score_a = [B, T, N, F]
        target_video = v - v_z0 # direction of the flow
        target_audio = a - a_z0 # direction of the flow

        loss_video = torch.square(score_v-target_video)
        loss_audio = torch.square(score_a-target_audio)

        loss_video = torch.mean(loss_video, dim=[2,3,4])
        loss_audio = torch.mean(loss_audio, dim=[2,3,4])

        #mask out the loss for the time steps that are greater than T

        loss_video = loss_video * (weigths)
        loss_video = torch.mean(loss_video) 

        loss_audio = loss_audio * (weigths)
        loss_audio = torch.mean(loss_audio)

        return {"loss": (loss_video + loss_audio)}

    def sample(self, model, v_z, a_z, model_kwargs, progress=True):
        B = v_z.shape[0]

        window_indexes = torch.linspace(0, self.window_size-1, steps=self.window_size, device=v_z.device).unsqueeze(0).repeat(B,1)


        # warm up with different number of warmup timestep to be more precise
        for i in tqdm(range(self.warmup_timesteps), disable=not progress):
            dt, t_partial, t_rf = self.calculate_prerolling_timestep(window_indexes, i)

            score_v, score_a = model(v_z, a_z, t_rf, **model_kwargs)

            v_z = v_z.detach().clone() + dt*score_v
            a_z = a_z.detach().clone() + dt*score_a

        v_f = v_z[:,0]
        a_f = a_z[:,0]

        v_z = torch.cat([v_z[:,1:], torch.randn_like(v_z[:,0]).unsqueeze(1)*self.noise_scale], dim=1)
        a_z = torch.cat([a_z[:,1:], torch.randn_like(a_z[:,0]).unsqueeze(1)*self.noise_scale], dim=1)

        def yield_frame():
            nonlocal v_z, a_z, window_indexes
            yield (v_f, a_f)

            dt = 1/(self.num_timesteps*self.window_size)

            while True:
                for i in range(self.num_timesteps):
                    tw = (self.num_timesteps - i)/self.num_timesteps
                    t = (window_indexes + tw)/self.window_size
                    t = 1-t

                    t_rf = t*(self.T-self.eps) + self.eps

                    score_v, score_a = model(v_z, a_z, t_rf, **model_kwargs)

                    v_z = v_z.detach().clone() + dt*score_v
                    a_z = a_z.detach().clone() + dt*score_a

                v = v_z[:,0]
                a = a_z[:,0]

                #remove the first element
                v_noise = torch.randn_like(v_z[:,0]).unsqueeze(1)*self.noise_scale
                a_noise = torch.randn_like(a_z[:,0]).unsqueeze(1)*self.noise_scale

                v_z = torch.cat([v_z[:,1:],v_noise], dim=1)
                a_z = torch.cat([a_z[:,1:],a_noise], dim=1)

                yield (v, a)
        
        return yield_frame
    
    def sample_a2v(self, model, v_z, a, model_kwargs, scale=1, progress=True):
        B = v_z.shape[0]
        window_indexes = torch.linspace(0, self.window_size-1, steps=self.window_size, device=v_z.device).unsqueeze(0).repeat(B,1)

        a_partial = a[:, :self.window_size]

        a_noise = torch.randn_like(a, device=v_z.device)*self.noise_scale
        a_noise_partial = a_noise[:, :self.window_size]


        with torch.enable_grad():
            # warm up with different number of warmup timestep to be more precise
            for i in tqdm(range(self.warmup_timesteps), disable=not progress):
                v_z = v_z.detach().requires_grad_(True)

                dt, t_partial, t_rf = self.calculate_prerolling_timestep(window_indexes, i)

                a_z = a_partial*t_partial + a_noise_partial*(1-t_partial)

                score_v, score_a = model(v_z, a_z, t_rf, **model_kwargs)

                loss = torch.square((a_partial-a_noise_partial)-score_a)
                grad = torch.autograd.grad(loss.mean(), v_z)[0]

                v_z = v_z.detach() + dt*score_v - ((t_partial+dt)!=1) * dt * grad * scale

        v_f = v_z[:,0].detach()
        v_z = torch.cat([v_z[:,1:], torch.randn_like(v_z[:,0]).unsqueeze(1)*self.noise_scale], dim=1)


        def yield_frame():
            nonlocal v_z, a, a_noise, window_indexes
            yield v_f

            dt = 1/(self.num_timesteps*self.window_size)

            while True:
                torch.cuda.empty_cache()

                a = a[:,1:]
                a_noise = a_noise[:,1:]
                if a.shape[1] < self.window_size:
                    a = torch.cat([a, torch.randn_like(a[:,0]).unsqueeze(1)*self.noise_scale], dim=1)
                    a_noise = torch.cat([a_noise, torch.randn_like(a[:,0]).unsqueeze(1)*self.noise_scale], dim=1)

                a_partial = a[:, :self.window_size]
                a_noise_partial = a_noise[:, :self.window_size]

                with torch.enable_grad():
                    for i in range(self.num_timesteps):
                        v_z = v_z.detach().requires_grad_(True)

                        tw = (self.num_timesteps - i)/self.num_timesteps
                        t = (window_indexes + tw)/self.window_size
                        t = 1-t

                        t_partial = t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                        t_rf = t*(self.T-self.eps) + self.eps

                        a_z = a_partial*t_partial + torch.randn_like(a_partial, device=v_z.device)*self.noise_scale*(1-t_partial)

                        score_v, score_a = model(v_z, a_z, t_rf, **model_kwargs)

                        loss = torch.square((a_partial-a_noise_partial)-score_a)
                        grad = torch.autograd.grad(loss.mean(), v_z)[0]

                        v_z = v_z.detach() + dt*score_v - ((t_partial+dt)!=1) * dt * grad * scale

                v = v_z[:,0].detach()

                v_noise = torch.randn_like(v_z[:,0]).unsqueeze(1)*self.noise_scale
                v_z = torch.cat([v_z[:,1:],v_noise], dim=1)
                yield v

        return yield_frame
    
    def sample_v2a(self, model, v, a_z, model_kwargs, scale=2, progress=True):
        B = a_z.shape[0]
        window_indexes = torch.linspace(0, self.window_size-1, steps=self.window_size, device=a_z.device).unsqueeze(0).repeat(B,1)

        v_partial = v[:, :self.window_size]
        v_noise = torch.randn_like(v, device=a_z.device)*self.noise_scale
        v_noise_partial = v_noise[:, :self.window_size]

        with torch.enable_grad():
            # warm up with different number of warmup timestep to be more precise
            for i in tqdm(range(self.warmup_timesteps), disable=not progress):
                a_z = a_z.detach().requires_grad_(True)

                dt, t_partial, t_rf = self.calculate_prerolling_timestep(window_indexes, i)

                v_z = v_partial*t_partial + v_noise_partial*(1-t_partial)

                score_v, score_a = model(v_z, a_z, t_rf, **model_kwargs)

                loss = torch.square((v_partial-v_noise_partial)-score_v)
                grad = torch.autograd.grad(loss.mean(), a_z)[0]

                a_z = a_z.detach() + dt*score_a - ((t_partial + dt)!=1) * dt * grad * scale

        a_f = a_z[:,0].detach()
        a_z = torch.cat([a_z[:,1:], torch.randn_like(a_z[:,0]).unsqueeze(1)*self.noise_scale], dim=1)

        def yield_frame():
            nonlocal v, v_noise, a_z, window_indexes
            yield a_f

            dt = 1/(self.num_timesteps*self.window_size)
            while True:
                torch.cuda.empty_cache()
                v = v[:,1:]
                v_noise = v_noise[:,1:]
                
                if v.shape[1] < self.window_size:
                    v = torch.cat([v, torch.randn_like(v[:,0]).unsqueeze(1)*self.noise_scale], dim=1)
                    v_noise = torch.cat([v, torch.randn_like(v[:,0]).unsqueeze(1)*self.noise_scale], dim=1)

                v_partial = v[:, :self.window_size]
                v_noise_partial = v_noise[:, :self.window_size]

                with torch.enable_grad():
                    for i in range(self.num_timesteps):
                        a_z = a_z.detach().requires_grad_(True)

                        tw = (self.num_timesteps - i)/self.num_timesteps
                        t = (window_indexes + tw)/self.window_size
                        t = 1-t

                        t_partial = t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                        t_rf = t*(self.T-self.eps) + self.eps

                        v_z = v_partial*t_partial + v_noise_partial*(1-t_partial)

                        score_v, score_a = model(v_z, a_z, t_rf, **model_kwargs)

                        loss = torch.square((v_partial-v_noise_partial)-score_v)
                        grad = torch.autograd.grad(loss.mean(), a_z)[0]

                        a_z = a_z.detach() + dt*score_a - ((t_partial + dt)!=1) * dt * grad * scale

                a = a_z[:,0].detach()

                a_noise = torch.randn_like(a_z[:,0]).unsqueeze(1)*self.noise_scale
                a_z = torch.cat([a_z[:,1:],a_noise], dim=1)


                yield a

        return yield_frame

    def calculate_prerolling_timestep(self, window_indexes, i):
        tw = (self.warmup_timesteps - i)/self.warmup_timesteps
        tw_future = (self.warmup_timesteps - (i+1))/self.warmup_timesteps

        t = window_indexes/self.window_size + tw
        
        #timestep for the next iteration, to calculate dt
        t_future = window_indexes/self.window_size + tw_future

        #Swap 0 with 1, 1 is full image, 0 is full noise
        t = 1-t
        t_future = 1 - t_future

        t = torch.clamp(t, 0, 1)
        t_future = torch.clamp(t_future, 0, 1)

        dt = torch.abs(t_future-t).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # [B, window_size, 1, 1, 1]

        t_partial = t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        t_rf= t*(self.T-self.eps) + self.eps
        return dt,t_partial,t_rf
    
    def get_z0(self, batch, train=True):

        if self.init_type == 'gaussian':
            ### standard gaussian #+ 0.5
            return torch.randn(batch.shape)*self.noise_scale
        else:
            raise NotImplementedError("INITIALIZATION TYPE NOT IMPLEMENTED") 