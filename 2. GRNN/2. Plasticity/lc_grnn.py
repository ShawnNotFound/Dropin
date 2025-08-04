import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LCFeatureExtractor(nn.Module):
    """
    Linear Cepstral Coefficients (LC) Feature Extractor
    """
    def __init__(self, n_fft=1024, hop_length=512, n_lc=20, sr=16000):
        super(LCFeatureExtractor, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_lc = n_lc
        self.sr = sr
        
        # Create DCT matrix for LC extraction
        self.register_buffer('dct_matrix', self._create_dct_matrix(self.n_fft // 2 + 1, self.n_lc))
        
    def _create_dct_matrix(self, n_bins, n_coeffs):
        """Create the DCT transform matrix"""
        dct_mat = torch.zeros(n_bins, n_coeffs)
        for k in range(n_coeffs):
            for n in range(n_bins):
                dct_mat[n, k] = torch.cos(torch.tensor(np.pi * k * (n + 0.5) / n_bins))
        
        # Normalize (except for first coefficient)
        dct_mat[:, 1:] = dct_mat[:, 1:] * torch.sqrt(torch.tensor(2.0 / n_bins))
        dct_mat[:, 0] = dct_mat[:, 0] * torch.sqrt(torch.tensor(1.0 / n_bins))
        
        return dct_mat
        
    def forward(self, waveform):
        """
        Extract LC features from waveform
        Args:
            waveform: Raw audio waveform (batch_size, 1, time)
        Returns:
            LC features (batch_size, n_lc, time)
        """
        batch_size = waveform.size(0)
        
        # Compute spectrogram
        spec = torch.stft(
            waveform.squeeze(1), 
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=torch.hann_window(self.n_fft).to(waveform.device),
            return_complex=True
        )
        
        # Compute power spectrogram
        pow_spec = torch.abs(spec).pow(2)
        
        # Apply log
        log_pow_spec = torch.log(pow_spec + 1e-6)
        
        # Apply DCT (LC computation)
        lc_features = torch.matmul(log_pow_spec.transpose(1, 2), self.dct_matrix)
        
        # Return features with proper shape (batch, n_lc, time)
        return lc_features.transpose(1, 2)


class GRNNLayer(nn.Module):
    """
    Gated Recurrent Neural Network (GRNN) Layer with dropin support
    """
    def __init__(self, input_size, hidden_size, dropin=False, dropin_size=0):
        super(GRNNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropin = dropin
        self.dropin_size = dropin_size
        
        # Original hidden size
        self.original_hidden_size = hidden_size
        
        # Create original layers
        # For update gate
        self.W_z_orig = nn.Linear(input_size, hidden_size)
        self.U_z_orig = nn.Linear(hidden_size, hidden_size)
        
        # For reset gate
        self.W_r_orig = nn.Linear(input_size, hidden_size)
        self.U_r_orig = nn.Linear(hidden_size, hidden_size)
        
        # For candidate activation
        self.W_h_orig = nn.Linear(input_size, hidden_size)
        self.U_h_orig = nn.Linear(hidden_size, hidden_size)
        
        # Create dropin layers if needed
        if dropin and dropin_size > 0:
            # For update gate - Note input is only from original part
            self.W_z_dropin = nn.Linear(input_size, dropin_size)
            # For U_z_dropin, input is concatenated [original_hidden, dropin_hidden]
            self.U_z_dropin_orig = nn.Linear(hidden_size, dropin_size)
            self.U_z_dropin_drop = nn.Linear(dropin_size, dropin_size)
            
            # For reset gate
            self.W_r_dropin = nn.Linear(input_size, dropin_size)
            self.U_r_dropin_orig = nn.Linear(hidden_size, dropin_size)
            self.U_r_dropin_drop = nn.Linear(dropin_size, dropin_size)
            
            # For candidate activation
            self.W_h_dropin = nn.Linear(input_size, dropin_size)
            self.U_h_dropin_orig = nn.Linear(hidden_size, dropin_size)
            self.U_h_dropin_drop = nn.Linear(dropin_size, dropin_size)
        
        # Initialize parameters
        self._init_parameters()
        
        # For tracking frozen status
        self.is_frozen = False
        
        # Total hidden size (original + dropin)
        self.total_hidden_size = hidden_size + (dropin_size if dropin and dropin_size > 0 else 0)
    
    def _init_parameters(self):
        """Initialize all parameters"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    
    def forward(self, x, h_prev=None):
        """
        Forward pass of GRNN
        Args:
            x: Input tensor (batch_size, input_size)
            h_prev: Previous hidden state (batch_size, total_hidden_size)
        Returns:
            h_new: New hidden state (batch_size, total_hidden_size)
        """
        batch_size = x.size(0)
        
        # Initialize hidden state if not provided
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.total_hidden_size, device=x.device)
        
        # Split the hidden state into original and dropin parts
        h_prev_orig = h_prev[:, :self.original_hidden_size]
        
        # Process through original layers
        # Update gate (original)
        z_orig = torch.sigmoid(self.W_z_orig(x) + self.U_z_orig(h_prev_orig))
        
        # Reset gate (original)
        r_orig = torch.sigmoid(self.W_r_orig(x) + self.U_r_orig(h_prev_orig))
        
        # Candidate activation (original)
        h_tilde_orig = torch.tanh(self.W_h_orig(x) + self.U_h_orig(r_orig * h_prev_orig))
        
        # New hidden state (original)
        h_new_orig = (1 - z_orig) * h_prev_orig + z_orig * h_tilde_orig
        
        # If dropin is enabled and has size > 0
        if self.dropin and self.dropin_size > 0:
            # Get dropin hidden state
            h_prev_dropin = h_prev[:, self.original_hidden_size:]
            
            # Update gate (dropin) - process original and dropin parts separately
            z_dropin = self.W_z_dropin(x) + self.U_z_dropin_orig(h_prev_orig) + self.U_z_dropin_drop(h_prev_dropin)
            z_dropin = torch.sigmoid(z_dropin)
            
            # Reset gate (dropin)
            r_dropin = self.W_r_dropin(x) + self.U_r_dropin_orig(h_prev_orig) + self.U_r_dropin_drop(h_prev_dropin)
            r_dropin = torch.sigmoid(r_dropin)
            
            # Create reset-applied hidden state for dropin
            r_h_prev_dropin = r_dropin * h_prev_dropin
            r_h_prev_orig = r_orig * h_prev_orig  # Using r_orig from the original calculation
            
            # Candidate activation (dropin)
            h_tilde_dropin = self.W_h_dropin(x) + self.U_h_dropin_orig(r_h_prev_orig) + self.U_h_dropin_drop(r_h_prev_dropin)
            h_tilde_dropin = torch.tanh(h_tilde_dropin)
            
            # New hidden state (dropin)
            h_new_dropin = (1 - z_dropin) * h_prev_dropin + z_dropin * h_tilde_dropin
            
            # Concatenate original and dropin hidden states
            h_new = torch.cat([h_new_orig, h_new_dropin], dim=1)
        else:
            h_new = h_new_orig
        
        return h_new


class LCGRNN(nn.Module):
    """
    LC-GRNN model with dropin support and SVM final layer
    """
    def __init__(self, input_size=20, hidden_sizes=[64, 32], dropin=False, dropin_sizes=[16, 8], num_classes=2):
        super(LCGRNN, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropin = dropin
        self.dropin_sizes = dropin_sizes if dropin else [0] * len(hidden_sizes)
        self.num_classes = num_classes
        
        # Validate dropin sizes
        if len(self.dropin_sizes) != len(self.hidden_sizes):
            raise ValueError("dropin_sizes must have the same length as hidden_sizes")
        
        # LC feature extractor
        self.lc_extractor = LCFeatureExtractor(n_lc=input_size)
        
        # GRNN layers
        self.grnn_layers = nn.ModuleList()
        
        # First layer takes input_size
        self.grnn_layers.append(
            GRNNLayer(input_size, hidden_sizes[0], dropin, self.dropin_sizes[0])
        )
        
        # Calculate total hidden sizes (original + dropin)
        total_hidden_sizes = [hidden_size + (dropin_size if dropin else 0) 
                             for hidden_size, dropin_size in zip(hidden_sizes, dropin_sizes)]
        
        # Remaining layers
        for i in range(1, len(hidden_sizes)):
            self.grnn_layers.append(
                GRNNLayer(total_hidden_sizes[i-1], hidden_sizes[i], dropin, self.dropin_sizes[i])
            )
        
        # SVM-like final layer
        self.svm_layer = nn.Linear(total_hidden_sizes[-1], num_classes)

    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Raw audio waveform (batch_size, 1, time)
        Returns:
            logits: Classification logits (batch_size, num_classes)
        """
        # Extract LC features
        features = self.lc_extractor(x)  # (batch_size, n_lc, time)
        
        # Process features through GRNN
        batch_size, _, seq_len = features.size()
        
        # For each time step
        hidden_states = [None] * len(self.grnn_layers)
        
        for t in range(seq_len):
            # Get features at this time step
            x_t = features[:, :, t]  # (batch_size, n_lc)
            
            # Pass through GRNN layers
            for i, layer in enumerate(self.grnn_layers):
                if i == 0:
                    hidden_states[i] = layer(x_t, hidden_states[i])
                else:
                    hidden_states[i] = layer(hidden_states[i-1], hidden_states[i])
        
        # Final hidden state from last layer
        final_hidden = hidden_states[-1]
        
        # SVM classifier
        logits = self.svm_layer(final_hidden)
        
        return logits

class SVMLoss(nn.Module):
    """
    SVM hinge loss with squared margin
    """
    def __init__(self, margin=1.0):
        super(SVMLoss, self).__init__()
        self.margin = margin
    
    def forward(self, outputs, targets):
        """
        Compute SVM loss
        Args:
            outputs: Model predictions (batch_size, num_classes)
            targets: Target classes (batch_size)
        Returns:
            loss: SVM loss value
        """
        batch_size = outputs.size(0)
        
        # Convert targets to one-hot
        targets_one_hot = F.one_hot(targets, num_classes=outputs.size(1)).float()
        
        # Compute margin loss
        correct_scores = torch.sum(outputs * targets_one_hot, dim=1)
        margins = outputs - correct_scores.unsqueeze(1) + self.margin
        margins = margins * (1 - targets_one_hot)  # Zero out the correct class
        
        # Hinge loss
        loss = torch.sum(F.relu(margins)) / batch_size
        
        return loss


# Create a custom optimizer wrapper that applies gradient masks
class MaskedOptimizer:
    """Wraps an optimizer to apply parameter masks before the optimization step"""
    def __init__(self, optimizer, model):
        self.optimizer = optimizer
        self.model = model
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def step(self):
        # Apply masks to gradients if the model supports it
        if hasattr(self.model, 'apply_mask_to_gradients'):
            self.model.apply_mask_to_gradients()
        
        # Perform the optimization step
        self.optimizer.step()