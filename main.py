import model
import utils
import numpy as np


if __name__ == "__main__":
	# ------------------------------ DATA -----------------------------------
	dataset = input("Enter dataset (FD001, FD002, FD003, FD004): ")
	# sensors to work with: T30, T50, P30, PS30, phi
	sensors = ['s_3', 's_4', 's_7', 's_11', 's_12']
	# sensors = ['s_{}'.format(i + 1) for i in range(0, 21)]
	# windows length
	sequence_length = 30
	# smoothing intensity
	alpha = 0.1
	# max RUL
	threshold = 125
	
	x_train, y_train, x_val, y_val, x_test, y_test = utils.get_data(dataset, sensors, 
	sequence_length, alpha, threshold)
	# -----------------------------------------------------------------------
	
	# ----------------------------- MODEL -----------------------------------
	# no_of_samples = x_train.shape[0]
	timesteps = x_train.shape[1]
	input_dim = x_train.shape[2]
	intermediate_dim = 300
	batch_size = 128
	latent_dim = 2
	epochs = 10000
	optimizer = 'adam'
	
	RVE = model.create_model(timesteps, 
			input_dim, 
			intermediate_dim, 
			batch_size, 
			latent_dim, 
			epochs, 
			optimizer,
			)
	
	# Callbacks for training
	model_callbacks = utils.get_callbacks(RVE, x_train, y_train)
	# -----------------------------------------------------------------------

	# --------------------------- TRAINING ---------------------------------
	results = RVE.fit(x_train, y_train,
			shuffle=True,
			epochs=epochs,
			batch_size=batch_size,
			validation_data= (x_val, y_val),
			callbacks=model_callbacks, verbose=2)
	# -----------------------------------------------------------------------

	# -------------------------- EVALUATION ---------------------------------
	RVE.load_weights('./checkpoints/checkpoint')
	train_mu = utils.viz_latent_space(RVE.encoder, np.concatenate((x_train, x_val)), np.concatenate((y_train, y_val)), save=True, show=False, tr_or_te='train')
	test_mu = utils.viz_latent_space(RVE.encoder, x_test, y_test.clip(upper=threshold), save=True, show=False, tr_or_te='test')
	# Evaluate
	y_hat_train = RVE.regressor.predict(train_mu)
	y_hat_test = RVE.regressor.predict(test_mu)

	utils.evaluate(np.concatenate((y_train, y_val)), y_hat_train, 'train')
	utils.evaluate(y_test, y_hat_test, 'test')
	# -----------------------------------------------------------------------
