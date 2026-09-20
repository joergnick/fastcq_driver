import scipy.io

res = scipy.io.loadmat("data/errorsh02cube.mat")
durations=   res["durations"]     
durationsf=  res["durationsf"]     
counters =   res["counters"]     
counterref =   res["counterref"]     
Ns       =   res["Ns"]     
err      =   res["err"]    
errf     =   res["errf"]   
errfvsn  =   res["errfvsn"]
print("Ns",Ns)
print("counters",counters)
print("counterref: ",counterref)
print("err",err)
print("errf",errf)
print("errfvsn",errfvsn)

print("durations",durations)
print("durationsf",durationsf)
