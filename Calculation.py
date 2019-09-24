# Python3.7 program  by BAHADUR Sk, Energy calculation
# Calculations 

def esult(energy, amplitude, time): 

	# Calculates Result 
	CI = energy * (pow((1 + amplitude / 100), time)) 
	print("Result is", CI) 

# Driver Code 
result(10000, 10.25, 5) 
