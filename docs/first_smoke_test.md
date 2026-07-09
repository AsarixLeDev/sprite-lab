baseline_496_smoke:
  train loss 4.699 -> 2.674 in 20 CPU steps
  val loss 2.653

baseline_496_overfit:
  train loss 4.730 -> 0.601 in 100 CPU steps on one batch
  val loss 7.496, expected due to one-batch overfit

eval reload:
  val loss 7.493, matches training eval
