import torch
from flash_attn import flash_attn_func


if __name__ == "__main__":
    for _ in range(64):
        # [batch_size, num_attention_heads, seq_len, head_dim]
        query = torch.rand(1, 32, 64, 64).bfloat16().cuda()
        # [batch_size, num_key_value_heads, seq_len, head_dim]
        key = torch.rand(1, 8, 64, 64).bfloat16().cuda()
        # [batch_size, num_key_value_heads, seq_len, head_dim]
        value = torch.rand(1, 8, 64, 64).bfloat16().cuda()

        flash_attn_outputs = flash_attn_func(
            q=query.transpose(-3, -2),
            k=key.transpose(-3, -2),
            v=value.transpose(-3, -2),
            causal=True,
        ).transpose(-3, -2)

        torch_attn_outputs = torch.nn.functional.scaled_dot_product_attention(
            query=query,
            key=key.repeat_interleave(4, dim=1),
            value=value.repeat_interleave(4, dim=1),
            is_causal=True,
        )

        print("Norm: ", torch.norm(flash_attn_outputs - torch_attn_outputs).item())

        assert torch.all(flash_attn_outputs == torch_attn_outputs).item()
