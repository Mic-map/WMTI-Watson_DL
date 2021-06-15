Inputs(6): md, ad, rd, mk, ak, rk
Output(6): f, Da, Depar, Deperp, kappa, f

data norm: True scale_type: 1, target scale: [100, 100, 100, 100, 1, 100] teacher_forcing_ratio: 0.3 lr: 0.0015

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
