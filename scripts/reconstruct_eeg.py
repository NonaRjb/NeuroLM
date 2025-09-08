import torch
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import os

from model.model_vq import VQ_Align
from dataset import PickleLoader
from model.model_neural_transformer import NTConfig

def std_norm(x):
    mean = torch.mean(x)
    std = torch.std(x)
    x = (x - mean) / std
    return x

@torch.no_grad()
def vq_reconstruct_tokens(model, X, input_chans, input_time, input_mask, device="cuda"):
    """
    X:            [B, N_tokens, 200]   (your dataset returns [block_size, 200]; add batch dim)
    input_chans:  [B, N_tokens]
    input_time:   [B, N_tokens]
    input_mask:   [B, N_tokens]  (bool)
    """
    model.eval()
    X = X.to(device)
    input_chans = input_chans.to(device)
    input_time  = input_time.to(device)
    input_mask  = input_mask.to(device)

    # Expand mask to 4D the same way your model.forward() does
    mask_4d = input_mask.unsqueeze(1).repeat(1, X.size(1), 1).unsqueeze(1)  # [B,1,N,N]

    # Encode → quantize → decode
    quantize, _, _, _ = model.VQ.encode(X, input_chans, input_time, mask_4d)
    _, xrec_raw = model.VQ.decode(quantize, input_chans, input_time, mask_4d)  # (rec_freq, rec_raw)

    # Return only valid tokens (strip masked/padded tail)
    valid_len = input_mask.sum(dim=1).item()  # assume B=1
    X_valid       = X[:, :valid_len, :]         # [1, V, 200]
    xrec_valid    = xrec_raw[:, :valid_len, :]  # [1, V, 200]
    chans_valid   = input_chans[:, :valid_len]  # [1, V]
    times_valid   = input_time[:, :valid_len]   # [1, V]
    return X_valid.squeeze(0).cpu(), xrec_valid.squeeze(0).cpu(), \
           chans_valid.squeeze(0).cpu(), times_valid.squeeze(0).cpu()

def plot_token_overlay(X_tok, Xrec_tok, title="Token overlay", output_dir="./output/reconstruction/"):
    # X_tok, Xrec_tok: [200]
    print(X_tok.shape)
    X_tok = std_norm(X_tok)
    t = torch.arange(X_tok.numel())
    plt.figure(figsize=(8, 3))
    plt.plot(t, X_tok.numpy(), label="Original", linewidth=1.2)
    plt.plot(t, Xrec_tok.numpy(), label="Reconstruction", linewidth=1.2, alpha=0.8)
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Std. amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "token_overlay.png"), dpi=300)
    # plt.show()

def stitch_tokens(tokens, times_valid, N_chans, hop_len=200, patch_len=200):
    """
    tokens:      [V, P]   V = W*N, P = patch_len
    times_valid: [V]      window index 0..W-1
    N_chans:               number of channels in the original sample
    Returns:
      cont: [N_chans, T_cont] continuous reconstruction by overlap-add with averaging.
    """
    V, P = tokens.shape
    print(f"Stitching {V} tokens of length {P} for {N_chans} channels with hop {hop_len}")
    W = int(times_valid.max().item()) + 1
    assert V % N_chans == 0, "V must be multiple of N_chans"
    # For each time window, there are N_chans tokens in row-major order as in your loader.
    # Rebuild per-channel lists
    T_cont = (W - 1) * hop_len + P
    print(f"Continuous length: {T_cont} samples")
    cont = torch.zeros((N_chans, T_cont))
    acc  = torch.zeros((N_chans, T_cont))

    # tokens are ordered as: for t in 0..W-1: for ch in 0..N-1: token
    # That is exactly how you created them via `list(ch_names) * time`
    # We can recover (t, ch) by:
    for v in range(V):
        t_idx = times_valid[v].item()
        ch_idx = v % N_chans  # matches construction in the loader
        start = t_idx * hop_len
        end   = start + P
        cont[ch_idx, start:end] += tokens[v]
        acc[ch_idx,  start:end] += 1.0

    acc = acc.clamp_min(1.0)
    cont = cont / acc
    return cont  # [N_chans, T_cont]

def pick_token_indices(chans_valid, times_valid, want_chan_idx=None, want_time=None):
    """
    chans_valid: [V] indices into standard_1020
    times_valid: [V] window indices (0..W-1)
    Returns the first matching token index.
    """
    idx = torch.arange(chans_valid.numel())
    keep = torch.ones_like(idx, dtype=torch.bool)
    if want_chan_idx is not None:
        keep = keep & (chans_valid == want_chan_idx)
    if want_time is not None:
        keep = keep & (times_valid == want_time)
    matches = idx[keep]
    return matches[0].item() if matches.numel() > 0 else None


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to dataset directory containing train/val/test subdirs with .pkl files')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint .pth file')
    parser.add_argument('--output_dir', type=str, default='./output/reconstruction/', help='Directory to save output plots')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'], help='Which data split to use')
    args = parser.parse_args()

    # --- load model + checkpoint ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # You must construct the model with the *same* encoder/decoder configs as training:
    # encoder_config, decoder_config = ... (load from your config file)

    P = 50
    H = 25
    # model init
    encoder_args = dict(n_layer=12, n_head=12, n_embd=768, block_size=1024,
                    bias=False, dropout=0., num_classes=0, in_chans=1, out_chans=16)
    decoder_args = dict(n_layer=4, n_head=12, n_embd=768, block_size=1024,
                    bias=False, dropout=0., num_classes=0, in_chans=128)

    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    checkpoint_model_args = checkpoint['encoder_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias']:
        encoder_args[k] = checkpoint_model_args[k]
    checkpoint_model_args = checkpoint['decoder_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias']:
        decoder_args[k] = checkpoint_model_args[k]
    # create the model
    encoder_conf = NTConfig(**encoder_args)
    decoder_conf = NTConfig(**decoder_args)
    model = VQ_Align(encoder_conf, decoder_conf, decoder_out_dim=P).to(device)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()

    # --- prepare one sample from your dataset ---
    files = Path(args.dataset_dir, args.split).rglob('*.pkl')
    files = [file for file in files]
    ds = PickleLoader(files, patch_size=P, overlap_size=H)
    # ds = PickleLoader(files=[...], block_size=1024, sampling_rate=200, GPT_training=False, hop_len_samples=200)  # or 100 if overlapped
    X, Y_freq, Y_raw, input_chans, input_time, input_mask = ds[1300]

    # Add batch dim
    X_b = X.unsqueeze(0)                    # [1, N, 200]
    input_chans_b = input_chans.unsqueeze(0)
    input_time_b  = input_time.unsqueeze(0)
    input_mask_b  = input_mask.unsqueeze(0)

    # --- reconstruct tokens ---
    X_valid, Xrec_valid, chans_valid, times_valid = vq_reconstruct_tokens(
        model, X_b, input_chans_b, input_time_b, input_mask_b, device=device
    )
    # X_valid, Xrec_valid: [V, 200]; chans_valid: [V]; times_valid: [V]

    # --- (A) Plot a single token overlay ---
    # Example: first token overall
    plot_token_overlay(X_valid[17], Xrec_valid[17], title="Token 0 (any channel/time)", output_dir=args.output_dir)

    # --- (B) Stitch continuous waveform and overlay for one channel ---
    N_chans = len(set(chans_valid.tolist()))  # or len(sample["ch_names"])
    hop_len = 25  # set 100 if you used 50% overlap at inference
    P = 50
    mu  = X_valid.mean()                 # over valid tokens&time
    std = X_valid.std().clamp_min(1e-6)
    X_std = (X_valid - mu) / std
    orig_cont = stitch_tokens(X_std,    times_valid, N_chans=N_chans, hop_len=hop_len, patch_len=P)
    reco_cont = stitch_tokens(Xrec_valid, times_valid, N_chans=N_chans, hop_len=hop_len, patch_len=P)

    # Pick a channel index to visualize (e.g., channel 0 in your file order)
    ch_to_plot = 25
    plt.figure(figsize=(10, 3))
    plt.plot(orig_cont[ch_to_plot].numpy(), label="Original (std. space)", linewidth=1.2)
    plt.plot(reco_cont[ch_to_plot].numpy(), label="Reconstruction", linewidth=1.2, alpha=0.85)
    plt.title(f"Channel {ch_to_plot} continuous overlay")
    plt.xlabel("Samples")
    plt.ylabel("Std. amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "continuous_overlay.png"), dpi=300)
    