import torch, os

from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
from torch.autograd import grad

NU = 0.01

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class InputFeatures(torch.nn.Module):
    """Base class for input features."""

    def __init__(self) -> None:
        super().__init__()
        self.outdim = None

class FourierFeatures(InputFeatures):
    '''
    Gaussian Fourier features, as proposed in Tancik et al., NeurIPS 2020.
    '''

    def __init__(self, scale, mapdim, indim) -> None:
        super().__init__()
        self.scale = scale
        self.mapdim = mapdim
        self.outdim = 2 * mapdim
        self.indim = indim

        B = torch.randn(self.mapdim, self.indim) * self.scale**2
        self.register_buffer('B', B)

    def forward(self, x):
        x_proj = (2. * torch.pi * x) @ self.B.T
        return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)
    
    def __repr__(self):
        return f"FourierFeatures(scale={self.scale}, mapdim={self.mapdim}, outdim={self.outdim})"



class PINN_Architecture(torch.nn.Module):
    def __init__(self, architecture, fourier_scale, fourier_mapdim, indim):
        super(PINN_Architecture, self).__init__()
        self.fourier_scale = fourier_scale
        self.fourier_mapdim = fourier_mapdim
        self.indim = indim
        self.layers = self._build_layers(architecture)

    def _build_layers(self, architecture):
        fourier_features = FourierFeatures(self.fourier_scale, self.fourier_mapdim, self.indim)
        model_layers = [fourier_features]
        model_layers.append(torch.nn.Linear(fourier_features.outdim, architecture[1]))
        # Add remaining layers
        for i in range(1, len(architecture)-1):
            model_layers.append(torch.nn.Softplus(beta=100))
            model_layers.append(torch.nn.Linear(architecture[i], architecture[i+1]))
        
        return torch.nn.Sequential(*model_layers)
    
    def forward(self, x):
        return self.layers(x)


class PINN_Raissi(torch.nn.Module):
    def __init__(self, df_train, bc_mask):
        super(PINN_Raissi, self).__init__()
        self.x = torch.tensor(df_train['x'].astype(float).values, requires_grad=True).float().unsqueeze(1).to(device)
        self.y = torch.tensor(df_train['y'].astype(float).values, requires_grad=True).float().unsqueeze(1).to(device)
    
        self.u = torch.tensor(df_train['u'].astype(float).values).float().unsqueeze(1).to(device)
        self.v = torch.tensor(df_train['v'].astype(float).values).float().unsqueeze(1).to(device)
        self.p = torch.tensor(df_train['p'].astype(float).values).float().unsqueeze(1).to(device)

        self.bc_mask = bc_mask

        self.psi_layers = [2, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 1]
        self.p_layers = [2, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 1]
        
        print(f"psi layers: {self.psi_layers}")
        print(f"p layers: {self.p_layers}")

        self.psi_model = self.create_model(self.psi_layers)
        self.p_model = self.create_model(self.p_layers)

        self.loss_func = torch.nn.MSELoss()

        self.lbfgs_optimizer_psi = torch.optim.LBFGS([{'params': self.psi_model.parameters()}], line_search_fn='strong_wolfe') 
        self.lbfgs_optimizer_p = torch.optim.LBFGS([{'params': self.p_model.parameters()}], line_search_fn='strong_wolfe') 

        self.writer = SummaryWriter(log_dir=f"logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}")    
        

    def create_model(self, type_layers):
        layers = []
        for i in range(len(type_layers) - 1):
            layers.append(torch.nn.Linear(type_layers[i], type_layers[i+1]))
            if i != len(type_layers) - 2:
                layers.append(torch.nn.Softplus(100))
        return torch.nn.Sequential(*layers)

    def net_NS(self, x, y):
        inputs = torch.cat([x, y], dim=1)
        psi = self.psi_model(inputs)
        p = self.p_model(inputs)
        
        u = grad(psi, y, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        v = -grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0]

        u_x = grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        v_x = grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        u_xx = grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        v_xx = grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_yy = grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
        p_x = grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

        f_u = (2 * u * u_x + u * v_y + v * u_y) + p_x - NU * (u_xx + u_yy)
        f_v = (2 * v * v_y + u * v_x + v * u_x) + p_y - NU * (v_xx + v_yy)
        
        return u, v, p, f_u, f_v    
    
    def forward(self, x, y):
        u_pred, v_pred, p_pred, f_u_pred, f_v_pred = self.net_NS(x, y)
        u_bc_loss = self.loss_func(self.u[self.bc_mask], u_pred[self.bc_mask])
        v_bc_loss = self.loss_func(self.v[self.bc_mask], v_pred[self.bc_mask])
        p_bc_loss = self.loss_func(self.p[self.bc_mask], p_pred[self.bc_mask])

        u_train_loss = self.loss_func(self.u, u_pred)
        v_train_loss = self.loss_func(self.v, v_pred)
        p_train_loss = self.loss_func(self.p, p_pred)

        rans_loss = self.loss_func(f_u_pred, torch.zeros_like(f_u_pred)) + self.loss_func(f_v_pred, torch.zeros_like(f_v_pred))
        psi_loss = u_bc_loss + v_bc_loss + rans_loss
        p_loss = p_bc_loss + rans_loss
        
        return psi_loss, p_loss, u_train_loss, v_train_loss, p_train_loss, rans_loss
    
    def train(self, nIter, checkpoint_path='path_to_checkpoint.pth'):
        self.display = {}
        self.temp_losses = {}
        loss_not_diminished_counter = 0
        last_loss = float('inf')

        if os.path.isfile(checkpoint_path):
            print(f"Loading checkpoint '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path)

            self.psi_model.load_state_dict(checkpoint['model_state_dict_psi'])
            self.lbfgs_optimizer_psi.load_state_dict(checkpoint['optimizer_state_dict_psi'])

            self.p_model.load_state_dict(checkpoint['model_state_dict_p'])
            self.lbfgs_optimizer_p.load_state_dict(checkpoint['optimizer_state_dict_p'])

            torch.set_rng_state(checkpoint['rng_state'])
            start_iteration = checkpoint.get('iterations', 0) + 1
            print(f"Resuming from iteration {start_iteration}")
        else:
            print(f"No checkpoint found at '{checkpoint_path}', starting from scratch.")
            start_iteration = 0

        def compute_losses():
            psi_loss, p_loss, u_train_loss, v_train_loss, p_train_loss, rans_loss = self.forward(self.x, self.y)
            self.display = {
                'psi_loss': psi_loss, 'p_loss': p_loss,
                'u_train_loss': u_train_loss, 'v_train_loss': v_train_loss, 'p_train_loss': p_train_loss,
                'rans_loss': rans_loss,
            }
            self.temp_losses = {'psi_loss': psi_loss, 'p_loss': p_loss,}

        for it in range(start_iteration, nIter + start_iteration):
            def closure_psi():
                self.lbfgs_optimizer_psi.zero_grad()
                compute_losses()
                self.temp_losses['psi_loss'].backward()
                return self.temp_losses['psi_loss']

            self.lbfgs_optimizer_psi.step(closure_psi)

            def closure_p():
                self.lbfgs_optimizer_p.zero_grad()
                compute_losses()
                self.temp_losses['p_loss'].backward()
                return self.temp_losses['p_loss']

            self.lbfgs_optimizer_p.step(closure_p)

            current_loss = self.temp_losses['psi_loss'].item() + self.temp_losses['p_loss'].item()
            if current_loss >= last_loss:
                loss_not_diminished_counter += 1
            else:
                loss_not_diminished_counter = 0
            last_loss = current_loss

            if loss_not_diminished_counter >= 10:
                print(f"Stopping early at iteration {it} due to no improvement.")
                break

            if it % 2 == 0: 
                print(f'It: {it}')
            if it % 10 == 0:
                for name, value in self.display.items():
                    print(f"{name}: {value.item()}")

                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)

                checkpoint = {
                    'model_state_dict_psi': self.psi_model.state_dict(),
                    'optimizer_state_dict_psi': self.lbfgs_optimizer_psi.state_dict(),

                    'model_state_dict_p': self.p_model.state_dict(),
                    'optimizer_state_dict_p': self.lbfgs_optimizer_p.state_dict(),

                    'iterations': it,

                    'rng_state': torch.get_rng_state(),
                }

                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint to '{checkpoint_path}' at iteration {it}")

    def predict(self, df_test):
        x_star = torch.tensor(df_test['x'].astype(float).values, requires_grad=True).float().unsqueeze(1).to(device)
        y_star = torch.tensor(df_test['y'].astype(float).values, requires_grad=True).float().unsqueeze(1).to(device)
        u_star, v_star, p_star, _, _ = self.net_NS(x_star, y_star)
        
        return u_star.cpu().detach().numpy(), v_star.cpu().detach().numpy(), p_star.cpu().detach().numpy()
