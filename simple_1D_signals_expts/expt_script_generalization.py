# To run: in folder the contains simple_1D_signals_expts, run:
#   python -m simple_1D_signals_expts.expt_script_generalization

from simple_1D_signals_expts.run import parse_args, main

args = parse_args()
args.ctx_sz = 10
args.fut_sz = 60
args.N = 1000
args.K = 10
args.global_seed = 2
args.pretrain_iters = 1000

args.num_random_models_to_plot = 15
args.num_top_k_models_to_plot = 3
args.plot_top_k = True
args.plot_top_1 = False
args.plot_random_models = True
args.plot_ensemble = True

# order: pretraining dataset, post-training dataset, test dataset
expts = [
          ['xavier', 'one_line', 'one_line'],
          ['kaiming', 'one_line', 'one_line'],
          ['ortho', 'one_line', 'one_line'],
          ['mixed', 'one_line', 'one_line'],
          ['line', 'one_line', 'one_line'],
          ['xavier', 'one_squarewave', 'one_squarewave'],
          ['kaiming', 'one_squarewave', 'one_squarewave'],
          ['ortho', 'one_squarewave', 'one_squarewave'],
          ['mixed', 'one_squarewave', 'one_squarewave'],
          ['line', 'one_squarewave', 'one_squarewave'],
          ['xavier', 'one_sinusoid', 'one_sinusoid'],
          ['kaiming', 'one_sinusoid', 'one_sinusoid'],
          ['ortho', 'one_sinusoid', 'one_sinusoid'],
          ['mixed', 'one_sinusoid', 'one_sinusoid'],
          ['line', 'one_sinusoid', 'one_sinusoid'],
          ['mixed', 'line', 'line'],
          ['mixed', 'sinusoid', 'sinusoid'],
          ['mixed', 'squarewave', 'squarewave'],
          ['sinusoid', 'squarewave', 'squarewave'],
          ['squarewave', 'squarewave', 'squarewave'],
]

for expt in expts:
  if expt[0] == 'xavier':
      args.pretrain_dataset = None
      args.sigma = 0.05
      args.base_init = 'xavier'
  elif expt[0] == 'kaiming':
      args.pretrain_dataset = None
      args.sigma = 0.05
      args.base_init = 'kaiming'
  elif expt[0] == 'ortho':
      args.pretrain_dataset = None
      args.sigma = 0.05
      args.base_init = 'ortho'
  else:
    args.sigma = 0.002
    args.pretrain_dataset = expt[0]
  args.posttrain_dataset = expt[1]
  args.test_dataset = expt[2]

  print(expt)
  main(args)
