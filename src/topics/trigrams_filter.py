import pickle 

trigrams_filter = [('does', 'not', 'have'),
]

trigrams_filter = [
('does', 'not', 'have'), 
('did', 'not', 'hav'), 
('do', 'not', 'have'), 
('can', 'not', 'have'), 
('have', 'not', 'been'), 
('do', 'not', 'know'), 
('do', 'not', 'like'), 
('do', 'not', 'feel'), 
('do', 'not', 'get'),
('do', 'not', 'leave'), 
('do', 'not', 'want'), 
('do', 'not', 'work'), 
('have', 'not', 'use')]


with open("trigrams_filter.txt", "wb") as f:
	pickle.dump(trigrams_filter, f)