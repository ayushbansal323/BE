

def power(a, b,P):
	if (b == 1) :
		return a; 

	else:
		return ((a**b) % P); 


 
def main():
	P = 23; 
	print("The value of P : "+ str(P)); 

	G = 9;  
	print("The value of G : "+ str(G)); 
 
	a = 4; 
	print("The private key a for Alice : "+ str(a)); 
	x = power(G, a, P); 
	

	b = 3;  
	print("The private key b for Bob : "+ str(b)); 
	y = power(G, b, P);  
 
	ka = power(y, a, P);  
	kb = power(x, b, P); 
	
	print("Secret key for the Alice is : "+ str(ka)); 
	print("Secret Key for the Bob is : "+ str(kb)); 
	
if __name__ == '__main__':
    main()
