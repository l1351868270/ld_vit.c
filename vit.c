
/*
gcc -o vit -g  vit.c -lm -fopenmp
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    int encoder_stride;
    int hidden_size;
    int image_size;
    int intermediate_size;
    int num_attention_heads;
    int num_channels;
    int num_hidden_layers;
    int patch_size;
    int num_labels;
} ViTConfig;

typedef struct {
    float *embeddings; // 
    float *eppw; // embeddings.patch_embeddings.projection.weight
    float *eppb; // vit.embeddings.patch_embeddings.projection.bias
    float *ec;   // embeddings.cls_token
    float *ep;   // embeddings.position_embeddings

    float *wq; // encoder.layer.{i}.attention.attention.query.weight
    float *bq; // encoder.layer.{i}.attention.attention.query.bias
    float *wk; // encoder.layer.{i}.attention.attention.key.weight
    float *bk; // encoder.layer.{i}.attention.attention.key.bias
    float *wv; // encoder.layer.{i}.attention.attention.value.weight
    float *bv; // encoder.layer.{i}.attention.attention.value.bias
    float *wd; // encoder.layer.{i}.attention.output.dense.weight
    float *bd; // encoder.layer.{i}.attention.output.dense.bias
    float *wdi; // encoder.layer.{i}.intermediate.dense.weight
    float *bdi; // encoder.layer.{i}.intermediate.dense.bias
    float *wdo; // encoder.layer.{i}.output.dense.weight
    float *bdo; // encoder.layer.{i}.output.dense.bias
    float *wlb; // encoder.layer.{i}.layernorm_before.weight
    float *blb; // encoder.layer.{i}.layernorm_before.bias
    float *wla; // encoder.layer.{i}.layernorm_after.weight
    float *bla; // encoder.layer.{i}.layernorm_after.bias

    float *wl; // layernorm.weight
    float *bl; // layernorm.bias
    float *wc; // classifier.weight
    float *bc; // classifier.bias
} ViTWeights;

typedef struct {
    float *ex;
    float *eh;
    float *eb;

    float *xb;
    float *xb2;
    float *hb;
    float *q;
    float *k;
    float *v;
    float *att;
    float *logits;
    int *logits_argmax;
    int N;
    int num_patches;
}RunState;

typedef struct {
    ViTConfig config; // the hyperparameters of the architecture (the blueprint)
    ViTWeights weights; 
    RunState state;
    float *params_memory;
    int num_parameters;
} ViT;

void malloc_run_state(RunState* s, ViTConfig* p) {
    s->ex = (float*)malloc(s->N * p->num_channels * p->image_size * p->image_size * sizeof(float));
    int H_out = (p->image_size - p->patch_size) / p->patch_size + 1;
    int W_out = (p->image_size - p->patch_size) / p->patch_size + 1;
    // int num_patches = H_out * W_out;
    int num_patches = s->num_patches;
    // printf("++++++++++++++++H_out:%d W_out:%d num_patches:%d\n", H_out, W_out, s->num_patches);
    s->eh = (float*)malloc(s->N * p->hidden_size * H_out * W_out * sizeof(float));
    s->eb = (float*)malloc(s->N * (num_patches + 1) * p->hidden_size * sizeof(float));

    s->xb = (float*)malloc(s->N * (num_patches + 1) * p->hidden_size * sizeof(float));
    s->xb2 = (float*)malloc(s->N * (num_patches + 1) * p->hidden_size * sizeof(float));

    s->hb = (float*)malloc(s->N * (num_patches + 1) * p->intermediate_size * sizeof(float));

    int all_head_size = p->num_attention_heads * (p->hidden_size / p->num_attention_heads);
    s->q = (float*)malloc(s->N * (num_patches + 1) * all_head_size * sizeof(float));
    s->k = (float*)malloc(s->N * (num_patches + 1) * all_head_size * sizeof(float));
    s->v = (float*)malloc(s->N * (num_patches + 1) * all_head_size * sizeof(float));

    s->att = (float*)malloc(s->N * p->num_attention_heads * (num_patches + 1) * (num_patches + 1) * sizeof(float));

    s->logits = (float*)malloc(s->N * p->num_labels * sizeof(float));
    s->logits_argmax = (int*)malloc(s->N * sizeof(int));
}
void free_run_state(RunState* s) {
    free(s->ex);
    free(s->eh);
    free(s->eb);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->logits_argmax);
}
void memory_map_weights(ViTWeights *w, ViTConfig* p, float* ptr) {
    w->eppw = ptr;
    ptr += p->hidden_size * p->num_channels * p->patch_size * p->patch_size;
    w->eppb = ptr;
    ptr += p->hidden_size;
    w->ec = ptr;
    ptr += p->hidden_size;
    w->ep = ptr;

    int num_patches = (p->image_size / p->patch_size) * (p->image_size / p->patch_size);
    ptr += (num_patches + 1 ) * p->hidden_size;
    w->wq = ptr;
    int all_head_size = p->num_attention_heads * (p->hidden_size / p->num_attention_heads);
    ptr += p->num_hidden_layers * p->hidden_size * all_head_size;
    w->bq = ptr;
    ptr += p->num_hidden_layers * all_head_size;
    w->wk = ptr;
    ptr += p->num_hidden_layers * p->hidden_size * all_head_size;
    w->bk = ptr;
    ptr += p->num_hidden_layers * all_head_size;
    w->wv = ptr;
    ptr += p->num_hidden_layers * p->hidden_size * all_head_size;
    w->bv = ptr;
    ptr += p->num_hidden_layers * all_head_size;
    w->wd = ptr;
    ptr += p->num_hidden_layers * p->hidden_size * p->hidden_size;
    w->bd = ptr;
    ptr += p->num_hidden_layers * p->hidden_size;
    w->wdi = ptr;
    ptr += p->num_hidden_layers * p->hidden_size * p->intermediate_size;
    w->bdi = ptr;
    ptr += p->num_hidden_layers * p->intermediate_size;
    w->wdo = ptr;
    ptr += p->num_hidden_layers * p->intermediate_size * p->hidden_size;
    w->bdo = ptr;
    ptr += p->num_hidden_layers * p->hidden_size;
    w->wlb = ptr;
    ptr += p->num_hidden_layers * p->hidden_size;
    w->blb = ptr;
    ptr += p->num_hidden_layers * p->hidden_size;
    w->wla = ptr;
    ptr += p->num_hidden_layers * p->hidden_size;
    w->bla = ptr;
    ptr += p->num_hidden_layers * p->hidden_size;

    w->wl = ptr;
    ptr += p->hidden_size;
    w->bl = ptr;
    ptr += p->hidden_size;
    w->wc = ptr;
    ptr += p->hidden_size * p->num_labels;
    w->bc = ptr;
}

void vit_build_from_checkpoint(ViT *model, char* checkpoint_path) {
    FILE *model_file = fopen(checkpoint_path, "rb");
    if (model_file == NULL) {
        printf("Error opening model file\n");
    }

    size_t file_size = 0;
    fseek(model_file, 0, SEEK_END);
    file_size = ftell(model_file);
    fseek(model_file, 0, SEEK_SET);
    printf("file_size is: %ld\n", file_size);

    int model_magic;
    fread(&model_magic, sizeof(int), 1, model_file);
    if (model_magic != 20240423) {
        printf("Bad magic model file\n");
    }
    printf("model magic is: %d\n", model_magic);

    fread(&model->config, sizeof(int), sizeof(model->config) / sizeof(int), model_file);
    printf("config encoder_stride is: %d\n", model->config.encoder_stride);
    printf("config hidden_size is: %d\n", model->config.hidden_size);
    printf("config image_size is: %d\n", model->config.image_size);
    printf("config intermediate_size is: %d\n", model->config.intermediate_size);
    printf("config num_attention_heads is: %d\n", model->config.num_attention_heads);
    printf("config num_channels is: %d\n", model->config.num_channels);
    printf("config num_hidden_layers is: %d\n", model->config.num_hidden_layers);
    printf("config patch_size is: %d\n", model->config.patch_size);
    printf("config num_labels is: %d\n", model->config.num_labels);
    
    int head_size = sizeof(model->config);
    size_t model_size = file_size - sizeof(model->config) - sizeof(int);

    model->num_parameters = model_size / sizeof(float);
    printf("num_parameters: %d\n", model->num_parameters);

    model->params_memory = (float*)malloc(model_size);
    fread(model->params_memory, sizeof(float), model->num_parameters, model_file);
    // for (int i = 0; i < 64; i++) {
    //     printf("weight: %f ", *(model->params_memory+i));
    // }
    model->weights.embeddings = model->params_memory;

    memory_map_weights(&model->weights, &model->config, model->params_memory);
}

typedef struct {

} Context;

typedef struct {
    // bchw
    int batch;
    int channel;
    int height;
    int width;
    float* data;
} Image;

void read_image(Image *img, char* img_path) {
    FILE *img_file = fopen(img_path, "rb");
    if (img_file == NULL) {
        printf("Error opening image file\n");
    }

    int headers[4];
    fread(headers, sizeof(int), 4, img_file);
    img->batch = headers[0];
    img->channel = headers[1];
    img->height = headers[2];
    img->width = headers[3];
    
    printf("image shape: %d %d %d %d\n", img->batch, img->channel, img->height, img->width);

    img->data = (float*)malloc(img->batch * img->channel * img->width * img->height * sizeof(float));
    fread(img->data, sizeof(float), img->batch * img->channel * img->width * img->height, img_file);
    // for (int i = img->batch * img->channel * img->height * img->width - 320; i < img->batch * img->channel * img->height * img->width; i++) {
    //     printf("%f ", *(img->data + i));
    // }
}

// https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
void conv2d_forward(float* output, float* input, float *kernel_weight, float* bias, 
                      int N, int C_in, int H_in, int W_in, int C_out, int H_out, int W_out, 
                      int kernel_size, int stride, int padding, int dilation, int is_bias, char* padding_mode) {
    if (stride < 0) {
        stride = 1;
    }
    printf("conv2d_forward N:%d C_in:%d H_in:%d W_in:%d C_out:%d H_out:%d W_out:%d kernel_size:%d stride:%d padding:%d\n", 
            N, C_in, H_in, W_in, C_out, H_out, W_out, 
            kernel_size,  stride, padding);
    for (int n = 0; n < N; n++) {
        int c_out = 0;
        #pragma omp parallel for private(c_out)
        for (c_out = 0; c_out < C_out; c_out++) {
            for (int h_in = 0; h_in < H_in + 2 * padding - kernel_size + 1; h_in += stride) {
                for (int w_in = 0; w_in < W_in + 2 * padding - kernel_size + 1; w_in += stride) {
                    int offset_out = n * C_out * H_out * W_out
                                   + c_out * H_out * W_out 
                                   + h_in / stride * W_out
                                   + w_in / stride;
                    float value = 0.0f;
                    for (int c_in = 0; c_in < C_in; c_in++) {
                        for (int k_i = 0; k_i < kernel_size; k_i++) {
                            for (int k_j = 0; k_j < kernel_size; k_j++){
                                int offset_kernel = c_out * C_in * kernel_size * kernel_size
                                                  + c_in * kernel_size * kernel_size
                                                  + k_i * kernel_size + k_j;
                                float input_v = 0.0f;
                                if (h_in + k_i >= padding && h_in + k_i < H_in + padding && w_in + k_j >= padding && w_in + k_j < W_in + padding) {
                                    int offset_in = n * C_in * H_in * W_in
                                                  + c_in * H_in * W_in
                                                  + (h_in - padding) * W_in
                                                  + (w_in - padding)
                                                  + k_i * W_in + k_j;
                                    input_v = input[offset_in];
                                
                                }
                                value += input_v * (*(kernel_weight + offset_kernel));
                            }
                        }                   
                    }
                    output[offset_out] = value;
                    if (is_bias != 0) {
                        output[offset_out] += bias[c_out];
                    }
                    
                    // if (offset_out < N * C_out * H_out * W_out && offset_out >= N * C_out * H_out * W_out - 640) {
                    // if (offset_out < 640) {
                    //     printf("conv2d_forwardV2 n:%d c_out:%d h_out:%d w_out:%d output[%d]: %f\n", n, c_out, h_in/stride, w_in/stride, offset_out, value);
                    // }
                }
            }
        }
    }
}

// https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
void linear_forward(float* output, float* input, float *weight, float* bias, int N, int L, int in_features, int out_features) {
    printf("linear_forward N:%d seq_len:%d in_features:%d out_features:%d\n", N, L, in_features, out_features);
    for (int n = 0; n < N; n++) {
        int l;
        #pragma omp parallel for private(l)
        for (l = 0; l < L; l++) {
            for(int out = 0; out < out_features; out++) {
                int offset_out = n * L * out_features + l * out_features + out;
                int offset_bias = out;
                float value = 0.0f;
                for (int in = 0; in < in_features; in++) {
                    int offset_in = n * L * in_features + l * in_features + in;
                    int offset_weight = out * in_features + in;
                    value += input[offset_in] * weight[offset_weight];

                }
                output[offset_out] = value + bias[offset_bias];
            }
        }
    }
}


void ViTEmbeddings(Context *ctx, float *output, float *input, RunState *s, ViTWeights *w, ViTConfig *p, Image *img, int B) {
    float *cls_token = w->ec;
    float *position_embeddings = w->ep;
    int H_out = (p->image_size - p->patch_size) / p->patch_size + 1;
    int W_out = (p->image_size - p->patch_size) / p->patch_size + 1;
    conv2d_forward(s->eh, input, w->eppw, w->eppb, 
                   B, p->num_channels, p->image_size, p->image_size, p->hidden_size, 
                   H_out, W_out, 
                    p->patch_size, p->patch_size, 0, 0, 1, NULL);
    // int num_patches = (p->image_size / p->patch_size) * (p->image_size / p->patch_size);
    int num_patches = H_out * W_out;
    int H = p->hidden_size;
    printf("ViTEmbeddings N:%d seq_len:%d hidden_size:%d\n", B, num_patches + 1, H);

    for (int n = 0; n < B; n++) {
        int l = 0;
        #pragma omp parallel for private(l)
        for (l = 0; l < num_patches + 1; l++) {
            for (int h = 0; h < H; h++) {
                int offset_position_embeddings = l * H + h;
                int offset_out = n * (num_patches + 1) * H + l * H + h;
                if (l == 0) {
                    int offset_cls_token = h;
                    output[offset_out] = cls_token[offset_cls_token] + position_embeddings[offset_position_embeddings];
                } else {
                    int offset_patch_embeddings = n * num_patches * H + (l-1) + h * num_patches;
                    output[offset_out] = s->eh[offset_patch_embeddings] + position_embeddings[offset_position_embeddings];
                }
            }
        }
    }
    // for (int i = B * (num_patches + 1) * H - 768 * 2; i < B * (num_patches + 1) * H; i++) {
    // // for (int i = 0; i < 768 * 2; i++) {
    //     printf("%d=%f ", i, output[i]);
    // }
}

// https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
void layernorm_forward(float* output, float* input, float *weight, float* bias, int N, int L, int features) {
    printf("layernorm_forward N:%d seq_len:%d dim:%d\n", N, L, features);
    for (int n = 0; n < N; n++) {
        int l = 0;
        #pragma omp parallel for private(l)
        for (l = 0; l < L; l++) {
            
            int offset = n * L * features + l * features;
            

            float mean = 0.0f;
            for (int f = 0; f < features; f++) {
                mean += input[offset + f];
            }

            mean /= features;
            float ss = 0.0f;
            for (int f = 0; f < features; f++) {
                ss += (input[offset + f] - mean) * (input[ offset + f] - mean);
            }
            ss /= features;
            ss += 1e-12f;
            ss = 1.0f / sqrtf(ss);
            
            for (int f = 0; f < features; f++) {
                 output[offset + f] = (input[offset + f] - mean) * ss * weight[f] + bias[f];
            }
            
        }
    }
}

void attention_forward(float* output, RunState *s, ViTWeights *w, ViTConfig *p, int seq_len) {
    int attention_head_size = p->hidden_size / p->num_attention_heads;
    int all_head_size = p->num_attention_heads * (p->hidden_size / p->num_attention_heads);
    printf("attention_forward N:%d seq_len:%d all_head_size:%d\n", s->N, seq_len, all_head_size);
    for (int n = 0; n < s->N; n++) {
        for (int h = 0; h < p->num_attention_heads; h++) {
            for (int lq = 0; lq < seq_len; lq++) {
                float *att = s->att + n * p->num_attention_heads* seq_len * seq_len 
                                        + h * seq_len * seq_len + lq * seq_len;
                float *q = s->q + n * seq_len * all_head_size 
                                + lq * all_head_size
                                + h * attention_head_size;

                for (int lk = 0; lk < seq_len; lk++) {
                    float *k = s->k + n * seq_len * all_head_size 
                                + lk * all_head_size
                                + h * attention_head_size;

                    float score = 0.0f;
                    for (int i = 0; i < attention_head_size; i++) {
                        score += q[i] * k[i];
                    }
                    score /= sqrtf(attention_head_size);
                    att[lk] = score;
                }
                float max_val = att[0];
                for (int lk = 1; lk < seq_len; lk++) { 
                    if (att[lk] > max_val) {
                        max_val = att[lk];
                    }
                }
                float ss = 0.0f;
                for (int lk = 0; lk < seq_len; lk++) { 
                    ss += expf(att[lk] - max_val);
                }

                for (int lk = 0; lk < seq_len; lk++) { 
                    att[lk] = expf(att[lk] - max_val) / ss;
                }
                
                float *o = output + n * seq_len * all_head_size
                         + lq * all_head_size
                         + h * attention_head_size;
                for (int lv = 0; lv < attention_head_size; lv++){
                    float sv = 0.0f;
                    for (int k = 0; k < seq_len; k++) { 
                        float *v = s->v + n * seq_len * all_head_size 
                                + k * all_head_size + lv
                                + h * attention_head_size;
                        sv += att[k] * (*v);
                    }
                    o[lv] = sv;
                }
            }
        }
    }
}

// https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
void gelu_forward(float* input, int N, int L, int features) {
    printf("gelu_forward N:%d seq_len:%d dim:%d\n", N, L, features);
    for (int n = 0; n < N; n++) {
        int l = 0;
        #pragma omp parallel for private(l)
        for (l = 0; l < L; l++) {
            for (int f = 0; f < features; f++) {
                int offset = n * L * features + l * features + f;
                // input[offset] = 0.5 * input[offset] * (1 + tanhf(sqrtf(2.0 / M_PI) * (input[offset] + 0.044715 * input[offset] * input[offset] * input[offset])));
                input[offset] = 0.5 * input[offset] * (1 + tanhf(sqrtf(2.0 / M_PI) * (input[offset] + 0.044715 * powf(input[offset], 3.0))));
            }
        }
    }
}

void softmax_forward(float* input, int M, int N) {
    printf("softmax_forward M:%d N:%d\n", M, N);
    int m;
    #pragma omp parallel for private(m)
    for (m = 0; m < M; m++) {
        float max_val = input[m * N];
        for (int n = 1; n < N; n++) {
            if (input[m * N + n] > max_val) {
                max_val = input[m * N + n];
            }
        }

        float sum = 0.0f;
        for (int n = 0; n < N; n++) {
            input[m * N + n] = expf(input[m * N + n] - max_val);
            sum += input[m * N + n];
        }
        for (int n = 0; n < N; n++) {
            input[m * N + n] /= sum;
        }
    }
}

void argmax_forward(int* output, float* input, int M, int N) {
    printf("argmax_forward M:%d N:%d\n", M, N);
    int m = 0;
    #pragma omp parallel for private(m)
    for (m = 0; m < M; m++) {
        int v = 0;
        for (int n = 1; n < N; n++) {
           if (input[m * N + n] > input[m * N + v]) {
               v = n;
           }
        }
        output[m] = v;
    }
}


void vit_forward(Context *ctx, ViT *model, Image* img, int B) {
    ViTConfig *p = &model->config;
    RunState* s = &model->state;
    ViTWeights *w = &model->weights;

    int num_patches = s->num_patches;
    int hidden_size = p->hidden_size;
    int intermediate_size = p->intermediate_size;
    int all_head_size = p->num_attention_heads * (p->hidden_size / p->num_attention_heads);
    

    if (p->num_channels != img->channel) {
        printf("Make sure that the channel dimension of the pixel values match with the one set in the configuration."
               "Expected %d but got %d. ", p->num_channels, img->channel);
    }
    if (img->height != p->image_size || img->width != p->image_size) {
        printf("Input image size (%d*%d) doesn't match model (%d*%d).", img->height, img->width, p->image_size, p->image_size);
    }
    int max_mem = B * p->hidden_size * (img->height / p->patch_size) * (img->width / p->patch_size) * sizeof(float);

    s->ex = (float*)malloc(img->batch * p->num_channels * p->image_size * p->image_size * sizeof(float));
    memcpy(s->ex, img->data, img->batch * p->num_channels * p->image_size * p->image_size * sizeof(float));
    ViTEmbeddings(ctx, s->eb, s->ex, s, w, p, img, B);

    
    // // for (int i = B * (num_patches + 1) * p->hidden_size - 768 * 2; i < B * (num_patches + 1) * p->hidden_size; i++) {
    // for (int i = 0; i < 768 * 2; i++) {
    //     printf("%d=%f ", i, s->eb[i]);
    // }

    for(int l = 0; l < p->num_hidden_layers; l++) {
        printf("+++++++++++++++++layer: %d\n", l);
    // for(unsigned long long l = 0; l < 3; l++) {
        layernorm_forward(s->xb, s->eb, w->wlb + l * hidden_size, w->blb + l * all_head_size, s->N, num_patches + 1, all_head_size);
        // // for (int i = B * (num_patches + 1) * all_head_size - 768 * 2; i < B * (num_patches + 1) * all_head_size; i++) {
        // for (int i = 0; i < 768 * 2; i++) {
        //     printf("%d=%f ", i, s->xb[i]);
        // }

        linear_forward(s->q, s->xb, w->wq + l*hidden_size*all_head_size, w->bq + l*all_head_size, s->N, num_patches+1, p->hidden_size, all_head_size);
        // // for (int i = B * (num_patches + 1) * all_head_size - 768 * 2; i < B * (num_patches + 1) * all_head_size; i++) {
        // for (int i = 0; i < 768 * 2; i++) {
        //     printf("%d=%f ", i, s->q[i]);
        // }
        linear_forward(s->k, s->xb, w->wk + l*hidden_size*all_head_size, w->bk + l*all_head_size, s->N, num_patches+1, p->hidden_size, all_head_size);
        // // for (int i = B * (num_patches + 1) * all_head_size - 768 * 2; i < B * (num_patches + 1) * all_head_size; i++) {
        // for (int i = 0; i < 768 * 2; i++) {
        //     printf("%d=%f ", i, s->k[i]);
        // }
        linear_forward(s->v, s->xb, w->wv + l*hidden_size*all_head_size, w->bv + l*all_head_size, s->N, num_patches+1, p->hidden_size, all_head_size);
        // for (int i = B * (num_patches + 1) * all_head_size - 768 * 2; i < B * (num_patches + 1) * all_head_size; i++) {
        // // for (int i = 0; i < 768 * 2; i++) {
        //     printf("%d=%f ", i, s->v[i]);
        // }
        // memset(s->xb, 0, B * (num_patches + 1) * all_head_size * sizeof(float));
        attention_forward(s->xb, s, w, p, num_patches + 1);
        // // for (int i = B * (num_patches + 1) * all_head_size - 768 * 2; i < B * (num_patches + 1) * all_head_size; i++) {
        // for (int i = 0; i < 768 * 2; i++) {
        //     printf("%d=%f ", i, s->xb[i]);
        // }

        // // for (int i = B * p->num_attention_heads * (num_patches + 1) * (num_patches + 1) - 197 * 2; i < B * p->num_attention_heads * (num_patches + 1) * (num_patches + 1); i++) {
        // for (int i = 0; i < 197 * 2; i++) {
        //     printf("%d=%f ", i, s->att[i]);
        // }
        linear_forward(s->xb2, s->xb, w->wd + l*hidden_size*hidden_size, w->bd + l*hidden_size, s->N, num_patches+1, p->hidden_size, p->hidden_size);
        // // for (int i = B * (num_patches + 1) * p->hidden_size - 768 * 2; i < B * (num_patches + 1) * p->hidden_size; i++) {
        // for (int i = 0; i < 768 * 2; i++) {
        //     printf("%d=%f ", i, s->xb2[i]);
        // }

        for (int i = 0; i < B * (num_patches + 1) * p->hidden_size; i++) {
            s->eb[i] += s->xb2[i];
        }
        // // for (int i = B * (num_patches + 1) * p->hidden_size - 768 * 2; i < B * (num_patches + 1) * p->hidden_size; i++) {
        // for (int i = 0; i < 768 * 2; i++) {
        //     printf("%d=%f ", i, s->eb[i]);
        // }
        // memset(s->xb, 0, B * (num_patches + 1) * all_head_size * sizeof(float));
        layernorm_forward(s->xb, s->eb, w->wla + l*hidden_size, w->bla + l*hidden_size, s->N, num_patches + 1, p->hidden_size);
        // // for (int i = B * (num_patches + 1) * all_head_size - 768 * 2; i < B * (num_patches + 1) * all_head_size; i++) {
        // for (int i = 0; i < 768 * 2; i++) {
        //     printf("%d=%f ", i, s->xb[i]);
        // }

        linear_forward(s->hb, s->xb, w->wdi + l*hidden_size*intermediate_size, w->bdi + l*intermediate_size, s->N, num_patches+1, p->hidden_size, p->intermediate_size);
        // // for (int i = B * (num_patches + 1) * p->intermediate_size - 3072; i < B * (num_patches + 1) * p->intermediate_size; i++) {
        // for (int i = 0; i < 3072; i++) {
        //     printf("%d=%f ", i, s->hb[i]);
        // }
        
        gelu_forward(s->hb, s->N, num_patches+1, p->intermediate_size);
        // // for (int i = B * (num_patches + 1) * p->intermediate_size - 3072; i < B * (num_patches + 1) * p->intermediate_size; i++) {
        // for (int i = 0; i < 3072; i++) {
        //     printf("%d=%f ", i, s->hb[i]);
        // }

        linear_forward(s->xb, s->hb, w->wdo + l*intermediate_size*hidden_size, w->bdo + l*hidden_size, s->N, num_patches+1, p->intermediate_size, p->hidden_size);
        // // for (int i = B * (num_patches + 1) * p->hidden_size - 768 * 2; i < B * (num_patches + 1) * p->hidden_size; i++) {
        // for (int i = 0; i < 768 * 2; i++) {
        //     printf("%d=%f ", i, s->xb[i]);
        // }

        for (int i = 0; i < B * (num_patches + 1) * p->hidden_size; i++) {
            s->eb[i] += s->xb[i];
        }
        // // for (int i = B * (num_patches + 1) * p->hidden_size - 768 * 2; i < B * (num_patches + 1) * p->hidden_size; i++) {
        // for (int i = 0; i < 768 * 2; i++) {
        //     printf("%d=%f ", i, s->eb[i]);
        // }
    }
    layernorm_forward(s->eb, s->eb, w->wl, w->bl, s->N, num_patches + 1, p->hidden_size);
    // for (int i = B * (num_patches + 1) * all_head_size - 768 * 2; i < B * (num_patches + 1) * all_head_size; i++) {
    // // for (int i = 0; i < 768 * 2; i++) {
    //     printf("%d=%f ", i, s->eb[i]);
    // }

    for (int i = 0; i < s->N; i++) {
        linear_forward(s->logits + i * p->num_labels, s->eb + i * (num_patches + 1) * p->hidden_size, w->wc, w->bc, 1, 1, p->hidden_size, p->num_labels);
    }
    // // // for (int i = B * (num_patches + 1) * p->hidden_size - 768 * 2; i < B * (num_patches + 1) * p->hidden_size; i++) {
    // for (int i = 0; i < s->N * p->num_labels; i++) {
    //     printf("%d=%f ", i, s->logits[i]);
    // }
    softmax_forward(s->logits, s->N, p->num_labels);
    // for (int i = 0; i < s->N * p->num_labels; i++) {
    //     printf("%d=%f ", i, *(s->logits + i));
    // }

    argmax_forward(s->logits_argmax, s->logits, s->N, p->num_labels);
    for (int i = 0; i < s->N; i++) {
        printf("image:%d, label id: %d prob:%f \n", i, *(s->logits_argmax + i), s->logits[*(s->logits_argmax + i)]);
    }
}

int main(int argc, char** argv) {
    ViT model;
    vit_build_from_checkpoint(&model, "vit-base-patch16-224.bin");

    Image img;
    read_image(&img, "image.bin");

    int B = 2;
    model.state.N = B;
    int num_patches = (model.config.image_size / model.config.patch_size) * (model.config.image_size / model.config.patch_size);
    model.state.num_patches = num_patches;

    malloc_run_state(&model.state, &model.config);
    Context ctx;
    vit_forward(&ctx, &model, &img, B);

    free_run_state(&model.state);
    return 0;
}