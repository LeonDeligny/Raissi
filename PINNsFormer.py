import torch, copy, os

from torch.autograd import grad
from torch.cuda.amp import autocast, GradScaler

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NU = 0.01

def get_clones(module, N):
    return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class WaveAct(torch.nn.Module): # Wavelet activation function
    def __init__(self):
        super(WaveAct, self).__init__() # Updates w1 and w2 during training
        self.w1 = torch.nn.Parameter(torch.ones(1), requires_grad=True) # Used in gradient computation during backpropagation
        self.w2 = torch.nn.Parameter(torch.ones(1), requires_grad=True) # w1 and w2 are learned parameters

    def forward(self, x):
        return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)

class FeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff=256):
        super(FeedForward, self).__init__() 
        self.linear = torch.nn.Sequential(*[
            torch.nn.Linear(d_model, d_ff),
            WaveAct(), # torch.nn.Softplus(beta=100), # WaveAct(),
            torch.nn.Linear(d_ff, d_ff),
            WaveAct(), # torch.nn.Softplus(beta=100), # WaveAct(),
            torch.nn.Linear(d_ff, d_model)
        ])

    def forward(self, x):
        return self.linear(x)

class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, heads):
        super(EncoderLayer, self).__init__()

        self.attn = torch.nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.ff = FeedForward(d_model)
        self.act1 = WaveAct() # torch.nn.Softplus(beta=100) # WaveAct()
        self.act2 = WaveAct() # torch.nn.Softplus(beta=100) # WaveAct()
        
    def forward(self, x):
        x2 = self.act1(x)
        x = x + self.attn(x2,x2,x2)[0]
        x2 = self.act2(x)
        x = x + self.ff(x2)
        return x

class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, heads):
        super(DecoderLayer, self).__init__()
        self.attn = torch.nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.ff = FeedForward(d_model)
        self.act1 = WaveAct() # torch.nn.Softplus(beta=100) # WaveAct()
        self.act2 = WaveAct() # torch.nn.Softplus(beta=100) # WaveAct()

    def forward(self, x, e_outputs): 
        x2 = self.act1(x)
        x = x + self.attn(x2, e_outputs, e_outputs)[0]
        x2 = self.act2(x)
        x = x + self.ff(x2)
        return x
    
class Encoder(torch.nn.Module):
    def __init__(self, d_model, N, heads):
        super(Encoder, self).__init__()
        self.N = N
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.act = WaveAct() # torch.nn.Softplus(beta=100) # WaveAct()

    def forward(self, x):
        for i in range(self.N):
            x = self.layers[i](x)
        return self.act(x)

class Decoder(torch.nn.Module):
    def __init__(self, d_model, N, heads):
        super(Decoder, self).__init__()
        self.N = N
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.act = WaveAct() # torch.nn.Softplus(beta=100) # WaveAct()
        
    def forward(self, x, e_outputs):
        for i in range(self.N):
            x = self.layers[i](x, e_outputs)
        return self.act(x)

class PINNsFormer_Architecture(torch.nn.Module):
    def __init__(self):
        super(PINNsFormer_Architecture, self).__init__()
        self.d_model = 64
        self.d_hidden = 1024

        self.linear_emb = torch.nn.Linear(3, self.d_model)
        self.encoder = Encoder(self.d_model, 2, 2)
        self.decoder = Decoder(self.d_model, 2, 2)
        self.linear_out = torch.nn.Sequential(*[
            torch.nn.Linear(self.d_model, self.d_hidden),
            WaveAct(), # torch.nn.Softplus(beta=100), # WaveAct(),
            torch.nn.Linear(self.d_hidden, self.d_hidden),
            WaveAct(), # torch.nn.Softplus(beta=100), # WaveAct(),
            torch.nn.Linear(self.d_hidden, 3)
        ])

class PINNsFormer(torch.nn.Module):
    def __init__(self, df, bc_mask, sym_mask, raissi_mask):
        super(PINNsFormer, self).__init__()
        self.x = torch.tensor(df['x'].astype(float).values, requires_grad=True).float().unsqueeze(1).to(device)
        self.y = torch.tensor(df['y'].astype(float).values, requires_grad=True).float().unsqueeze(1).to(device)
        self.sdf = torch.tensor(df['sdf'].astype(float).values).float().unsqueeze(1).to(device)

        self.u = torch.tensor(df['u'].astype(float).values).float().unsqueeze(1).to(device)
        self.v = torch.tensor(df['v'].astype(float).values).float().unsqueeze(1).to(device)
        self.p = torch.tensor(df['p'].astype(float).values).float().unsqueeze(1).to(device)

        self.bc_mask = bc_mask
        self.sym_mask = sym_mask
        self.raissi_mask = raissi_mask

        self.model = PINNsFormer_Architecture()
        self.model.apply(init_weights)
        self.mse_loss = torch.nn.MSELoss()
        self.scaler = GradScaler()

        self.lbfgs_optimizer = torch.optim.LBFGS([{'params': self.model.parameters()}], line_search_fn='strong_wolfe') 

    def net_NS(self):
        src = torch.cat((self.x, self.y, self.sdf), dim=-1) # self.sdf
        src = self.model.linear_emb(src)
        e_outputs = self.model.encoder(src)
        d_output = self.model.decoder(src, e_outputs)
        output = self.model.linear_out(d_output)
        u = output[:, 0:1]
        v = output[:, 1:2]
        p = output[:, 2:3]

        # Calculate gradients for u and drop
        u_x = grad(u, self.x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = grad(u, self.y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = grad(u_x, self.x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = grad(u_y, self.y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        p_x = grad(p, self.x, grad_outputs=torch.ones_like(p), create_graph=True)[0]

        f_u = (v * u_y) + (u * u_x) + p_x - NU * (u_xx + u_yy)
        del u_y, u_xx, u_yy, p_x  # Drop the gradients no longer needed

        # Calculate gradients for v and drop
        v_x = grad(v, self.x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = grad(v, self.y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_xx = grad(v_x, self.x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_yy = grad(v_y, self.y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
        p_y = grad(p, self.y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

        f_v = (u * v_x) + (v * v_y) + p_y - NU * (v_xx + v_yy)
        del v_x, v_xx, v_yy, p_y  # Drop the gradients no longer needed

        ic = u_x + v_y 
        
        return u, v, p, f_u, f_v, ic
    
    def forward(self):
        u_pred, v_pred, p_pred, f_u_pred, f_v_pred, ic_pred = self.net_NS()
        u_bc_loss = self.mse_loss(self.u[self.bc_mask], u_pred[self.bc_mask])
        v_bc_loss = self.mse_loss(self.v[self.bc_mask], v_pred[self.bc_mask])
        p_bc_loss = self.mse_loss(self.p[self.bc_mask], p_pred[self.bc_mask])

        u_train_loss = self.mse_loss(self.u[raissi_mask], u_pred[raissi_mask])
        v_train_loss = self.mse_loss(self.v[raissi_mask], v_pred[raissi_mask])
        p_train_loss = self.mse_loss(self.p[raissi_mask], p_pred[raissi_mask])
        
        u_sym_loss = self.mse_loss(u_pred[self.sym_mask], u_pred[~self.sym_mask])
        p_sym_loss = self.mse_loss(p_pred[self.sym_mask], p_pred[~self.sym_mask])

        rans_loss = self.mse_loss(f_u_pred, torch.zeros_like(f_u_pred)) + self.mse_loss(f_v_pred, torch.zeros_like(f_v_pred))
        ic_loss = self.mse_loss(ic_pred, torch.zeros_like(ic_pred))
        loss = (u_bc_loss + u_sym_loss + v_bc_loss + p_bc_loss + p_sym_loss) + 2*(rans_loss + ic_loss)
        
        return loss, u_train_loss, v_train_loss, p_train_loss, u_bc_loss, v_bc_loss, rans_loss, ic_loss

    
    def train(self, nIter, checkpoint_path='path_to_checkpoint.pth'):
        self.display = {}
        self.temp_losses = {}
        loss_not_diminished_counter = 0
        last_loss = float('inf')

        start_iteration = self.load_checkpoint(checkpoint_path)

        def compute_losses():
            loss, u_train_loss, v_train_loss, p_train_loss, u_bc_loss, v_bc_loss, rans_loss, ic_loss = self.forward()
            self.display = {
                'u_train_loss': u_train_loss, 'v_train_loss': v_train_loss, 'p_train_loss': p_train_loss,
                'u_bc_loss': u_bc_loss, 'v_bc_loss': v_bc_loss,
                'rans_loss': rans_loss, 'ic_loss': ic_loss,
            }
            self.temp_losses = {'loss': loss}

        for it in range(start_iteration, nIter + start_iteration):
            def closure():
                self.lbfgs_optimizer.zero_grad()
                with autocast():
                    compute_losses()
                # self.temp_losses['loss'].backward()
                self.scaler.scale(self.temp_losses['loss']).backward()
                return self.temp_losses['loss']
            
            loss = self.scaler.scale(closure())  # Note: call scaler.scale() on the closure
            self.scaler.step(self.lbfgs_optimizer)
            self.scaler.update()

            # self.lbfgs_optimizer.step(closure)

            current_loss = self.temp_losses['loss'].item()
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
                self.save_checkpoint(checkpoint_path, it)

    def predict(self):
        u_star, v_star, p_star, f_u_star, f_v_star, ic_star = self.net_NS()
        
        return u_star.cpu().detach().numpy(), v_star.cpu().detach().numpy(), p_star.cpu().detach().numpy(), f_u_star.cpu().detach().numpy(), f_v_star.cpu().detach().numpy(), ic_star.cpu().detach().numpy()

    def load_checkpoint(self, checkpoint_path): 
        if os.path.isfile(checkpoint_path):
            print(f"Loading checkpoint '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.lbfgs_optimizer.load_state_dict(checkpoint['lbfgs_optimizer_state_dict'])
                        
            # Restore the RNG state
            torch.set_rng_state(checkpoint['rng_state'])

            # If you're resuming training and want to start from the next iteration,
            # make sure to load the last iteration count and add one
            start_iteration = checkpoint.get('iterations', 0) + 1
            print(f"Resuming from iteration {start_iteration}")
        else:
            print(f"No checkpoint found at '{checkpoint_path}', starting from scratch.")
            start_iteration = 0

        return start_iteration

    def save_checkpoint(self, checkpoint_path, it):
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'lbfgs_optimizer_state_dict': self.lbfgs_optimizer.state_dict(),

            'iterations': it,

            'rng_state': torch.get_rng_state(),
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to '{checkpoint_path}' at iteration {it}")

    
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)