# RngPredictor
Predicts xorshift128

You should just be able to pull and run `trainer.py` with `tensorflow-nightly` installed.  Note that it will go through the relatively slow process of hyperparameter tuning (the HP logfiles are too big and I don't know how to separate them out) so I recommend manually overwriting the hyperparameters with the ones from `results` if you don't want to do that search. Or you can just load the `weights` file and build from there :)
