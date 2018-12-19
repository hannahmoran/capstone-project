import pickle 

bigrams_filter = [('does', 'not', 'have'),
]

bigrams_filter = [
('does', 'not'), 
('did', 'not'), 
('do', 'not'), 
('can', 'not'), 
('have', 'not'), 
('have', 'been'), 
('am', 'not'), 
('is', 'not'), 
('are', 'not'), 
('would', 'be'),
('would', 'not'), 
('was', 'not'), 
('not', 'know'), 
('are', 'good'),
('be', 'not'), 
('be','be'),]


with open("bigrams_filter.txt", "wb") as f:
	pickle.dump(bigrams_filter, f)