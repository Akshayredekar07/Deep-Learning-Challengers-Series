### **Self-Attention Using Scaled Dot-Product Approach**

![alt text](images/image-2.png)
![alt text](images/image-3.png)
![alt text](images/image-4.png)
![alt text](images/image-5.png)
![alt text](images/image-6.png)
![alt text](images/image-7.png)

### 💠**The goal of Attention Mechanism to build context similarity matrix**

![alt text](images/image-8.png)
![alt text](images/image-9.png)
![alt text](images/image-10.png)
![alt text](images/image-11.png)
![alt text](images/image-12.png)
![alt text](images/image-13.png)
![alt text](images/image-14.png)
![alt text](images/image-15.png)
![alt text](images/image-16.png)
![alt text](images/image-17.png)
![alt text](images/image-18.png)
![alt text](images/image-19.png)
![alt text](images/image-20.png)
![alt text](images/image-21.png)

### **A Dive Into Multihead Attention, Self-Attention and Cross-Attention**
![alt text](images/image.png)
![alt text](images/image-1.png)
![alt text](images/image-22.png)
![alt text](images/image-23.png)
![alt text](images/image-24.png)
![alt text](images/image-25.png)
![alt text](images/image-26.png)
![alt text](images/image-27.png)

**Multihead Attention**
![alt text](images/image-28.png)
![alt text](images/image-29.png)
![alt text](images/image-30.png)
![alt text](images/image-31.png)
![alt text](images/image-32.png)

**Cross Attention**
![alt text](images/image-33.png)
![alt text](images/image-34.png)
![alt text](images/image-35.png)
![alt text](images/image-36.png)
![alt text](images/image-37.png)

```
Solution of the exercise:
We have 
X: T1xd
Y: T2xd

So, we build Q from Y, so that means Q will be 
Q: T2xd

And we build K and V from X, therefore,
K: T1xd
V: T1xd

Then, QK^t (compatibility matrix) will be 
QK^t: T2xT1

And the final output Z, will be Softmax(1/sqrt(d) QK^t) * V
Z: T2xd
```