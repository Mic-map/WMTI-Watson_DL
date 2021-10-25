input(6): md, ad, rd, mk, ak, rk

output(5): f, Da, Depar, Deperp, kappa

data norm: True

input scale type: 3, min_max_scale to [[0,3], [0,3], [0,3], [0, 10], [0, 10], [0, 10]], scale=100
output scale type: 2, min_max_scale to [[0, 1], [0, 3], [0, 3], [0, 3], [0, 1]], scale=100

teacher_forcing_ratio: 0.3

lr: 0.001

EncoderRNN(
  (lstm): LSTM(1, 96, batch_first=True)
  (bn): BatchNorm1d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (layer_norm): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
)

AttnDecoderRNN(
  (attn): Linear(in_features=193, out_features=6, bias=True)
  (attn_combine): Linear(in_features=97, out_features=96, bias=True)
  (bn): BatchNorm1d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (layer_norm): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
  (lstm): LSTM(96, 96, batch_first=True)
  (dropout1): Dropout(p=0, inplace=False)
  (dropout2): Dropout(p=0, inplace=False)
  (out): Linear(in_features=96, out_features=1, bias=True)
)
