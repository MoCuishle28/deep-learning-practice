

def compute_imp(base, target):
	for x in base:
		print( round((((target - x)/x)) * 100, 2) )

######################### RC15 #########################################

base_hr5 = [0.3054, 0.3169, 0.3124]
best_hr5 = 0.3365
# compute_imp(base_hr5, best_hr5)

base_ndcg5 = [0.2102, 0.219, 0.2167]
best_ndcg5 = 0.2356
# compute_imp(base_ndcg5, best_ndcg5)

base_hr10 = [0.4003, 0.4152, 0.41]
best_hr10 = 0.4291
# compute_imp(base_hr10, best_hr10)

base_ndcg10 = [0.2411, 0.2509, 0.2483]
best_ndcg10 = 0.2657
# compute_imp(base_ndcg10, best_ndcg10)

base_hr20 = [0.4807, 0.4963, 0.4884]
best_hr20 = 0.5034
# compute_imp(base_hr20, best_hr20)

base_ndcg20 = [0.2615, 0.2715, 0.2682]
best_ndcg20 = 0.2846
# compute_imp(base_ndcg20, best_ndcg20)

while True:
	base_data = input()
	base_data = [float(x) for x in base_data.split(',')]
	best_data = base_data.pop()
	compute_imp(base_data, best_data)
	print('---')