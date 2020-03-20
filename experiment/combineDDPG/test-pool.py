import multiprocessing


num_thread = 2


def func(args):
	return args[0] + args[1]

if __name__ == '__main__':
	args = [[i, i+1] for i in range(20)]
	
	pool = multiprocessing.Pool(processes=num_thread)

	res = pool.map(func, args)
	pool.close()
	pool.join()

	for x in res:
		print(x)