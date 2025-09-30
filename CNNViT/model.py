# define CNN model
import torch
import torch.nn as nn
from utils import Params
from torchinfo import summary

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.TransformerEncoderLayer)):
        if hasattr(module,'weight'):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity='relu')
        if hasattr(module,'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)

# split each image into patches
class PatchEmbedding(nn.Module):
    def __init__(self, img_W, img_H, kernel_size, embedding_dim):
        super().__init__()
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):                    #since we have a size of (width*height) for both image and patch
                return (x, x)
            return x
        img_size, kernel_size = (img_W, img_H), _make_tuple(kernel_size)
        self.num_patches = (img_size[0] // kernel_size[0]) * (
            img_size[1] // kernel_size[1])
        self.conv = nn.Conv2d(1, embedding_dim, kernel_size=kernel_size,
                                  stride=kernel_size)

    def forward(self, X):
        # Output shape: (batch size, no. of patches, no. of channels)
        return self.conv(X).flatten(2).transpose(1, 2)


# define the vision transformer
class MyCNNViT(nn.Module):
    def __init__(self, params):
        super().__init__()
        # tensors obtained from resized matrices
        self.patch_embedding = PatchEmbedding(params.img_W, params.img_H, params.kernel_size, params.embedding_dim)           # output size:  224//14  and  224//14   so 16*16=256 number of batches
        
        # Embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, params.embedding_dim) )        # cls token is introduced as a learnable parameter
        num_steps = self.patch_embedding.num_patches + 1  # Add the cls token
        # Positional embeddings are learnable
        self.pos_embedding = nn.Parameter(torch.randn(1, num_steps, params.embedding_dim))
        self.dropout = nn.Dropout(params.drop_out_stoch)
        
        # Encoder layer 
        encoder_layer = nn.TransformerEncoderLayer(d_model=params.embedding_dim, nhead=params.num_heads, dropout=params.drop_out_att, dim_feedforward=int(params.embedding_dim*2), activation="gelu", batch_first=True)
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer, num_layers=params.num_blks)
        
        # last MLP (head)
        self.head = nn.Sequential(nn.LayerNorm(params.embedding_dim), nn.Linear(params.embedding_dim, 2))
        
    def forward(self,x):
        y_patch_embedding = self.patch_embedding(x)
        y_concat = torch.cat((self.cls_token.expand(y_patch_embedding.shape[0], -1, -1), y_patch_embedding), 1)   # change the token size to the size of the batches. Then add the token class to the begining of each batch
        y_dropout = self.dropout(y_concat + self.pos_embedding)
        y_encoder = self.encoder_blocks(y_dropout)

        return self.head(y_encoder[:,0])     # extract the first token (which is the [CLS] token) 



def main():
    torch.manual_seed(42)
    params = Params(r'CNNViT/parameters/my_params.json')
    sample_input = torch.randn(2, 1, params.img_W, params.img_H)

    # # check patchembedding output
    patch_emb = PatchEmbedding(params.img_W, params.img_H, params.kernel_size, params.embedding_dim)
    print('number of patches:', patch_emb.num_patches)
    out = patch_emb(sample_input)
    print(out.shape)

    # check model output
    model = MyCNNViT(params)
    model.apply(init_weights)
    output = model(sample_input)                # output_shape = batch_size,    number_of_patches = ((H//batch_size)*(W//batch_size)),     embedding dimension
    print(output.shape)
    print(summary(model.to('cuda'), input_size=(1, 1, params.img_W, params.img_H)))
    

if __name__ == '__main__':
    main() 