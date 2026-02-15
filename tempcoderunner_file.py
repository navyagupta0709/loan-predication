test = [ 4887, 0, 133, 360, 2, 6, 0, 4, 3, 1, 5 ]
import numpy as np 
test2 = np.array(test).reshape(1,-1)
print(model.predict(test2))
