msg = input("Enter the Message:")
r = 0

# countining th redundancy bits

while (pow(2, r) < len(msg) + r + 1):
    r = r + 1
n = r + len(msg)
print("the original message with redundancy bits is :", n)
print("r value is ", r)

#  -------------------------------

# creating table

l1 = ['a'] * n
for i in range(r):
    l1[pow(2, i) - 1] = 'r'
l1.reverse()
for j in range(0, n):
    if l1[j] == 'a':
        l1[j] = msg[0]
        msg = msg[1:]
# print(l1)

#  -------------------------------


# creating decimal values for each location


pairstr = ''
pairlst = []
for i in range(n):
    pairstr = str(bin(i + 1)[2:])
    if len(pairstr) != r:
        for _ in range(r - len(pairstr)):
            pairstr = '0' + pairstr
    pairlst.append(pairstr)
# print(pairlst)

#  -------------------------------

# creating strings for pairty bits

finalpairty = []
reference = []
for i in range(r):
    finalstr = ''
    reference_sub = []
    for j in range(n):
        if pairlst[j][-(i + 1)] == '1':
            reference_sub.append(-(j + 1))
            if l1[-(j + 1)] != 'r':
                finalstr = finalstr + l1[-(j + 1)]
    finalpairty.append(finalstr)
    reference.append(reference_sub)
# print(finalpairty)
# print(reference) # this is used as reference in reciver side

#  -------------------------------

# caluculating the even pairty


even_pairty = []

for i in range(len(finalpairty)):
    if finalpairty[i].count('1') % 2 == 0:
        even_pairty.append('0')
    else:
        even_pairty.append('1')
# print(even_pairty)

#  -------------------------------

# chainging bits in list according to even pairty

for i in range(n):
    if l1[-i] == 'r':
        l1[-i] = even_pairty[0]
        even_pairty.pop(0)
#  -------------------------------


print("the message from the sender is : ", *l1)  # printing the table

rec = input("Enter the recived data : ")  # scanning recivers data
rec_list = [*rec]

# print(rec_list)
# print(reference)
# if the message is same as reciver we will say no error

if l1 == list(rec_list):
    print("no error")

#  -------------------------------

# else we will find error location

else:
    pairty_str = ''
    # rec_val =0
    for i in range(len(reference)):
        rec_str = ''
        for j in range(len(reference[i])):
            rec_str = rec_str + str(rec_list[reference[i][j]])
        if rec_str.count('1') % 2 == 0:
            pairty_str = pairty_str + '0'
        else:
            pairty_str = pairty_str + '1'
    # print(pairty_str)
    x = str(pairty_str)
    a = x[::-1]
    rec_val = int(a, 2)

    # print(a)
    print(" error is at location : ", (n - rec_val) + 1)