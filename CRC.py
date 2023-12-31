def divisionEx_OR(divisor, dividend):
    divisor_len = len(divisor)
    temp_remind = dividend[:divisor_len]

    # performing devision using Ex-OR operation.....
    while (True):
        temp = ''
        for i in range(divisor_len):  # Ex-OR operation
            if divisor[i] == '0' and temp_remind[i] == '0':
                temp += '0'
            elif divisor[i] == '1' and temp_remind[i] == '1':
                temp += '0'
            else:
                temp += '1'

        while (temp):  # removing 0's from starting of the reminder
            if temp[0] == '0':
                temp = temp[1:]
            else:
                break

        if (len(dividend) + len(temp) >= divisor_len):
            while (len(temp) < divisor_len):  # for adding 0/1's from dividend to temp_reminder
                temp += dividend[0]
                dividend = dividend[1:]
        else:
            reminder = temp
            break
        temp_remind = temp
    return reminder

mess = input("Enter the message for sending to reciever : ")
parity_gen = input("Enter the parity generato : ")


for i in range(len(parity_gen) - 1):
    mess += '0'
print("adding 0's to the messege : ", mess)

# changing var_names......
divisor = parity_gen;
dividend = mess;

reminder = divisionEx_OR(divisor, dividend)
print("reminder after X-OR (sender) : ", reminder)


# for replacing 0's by reminder that are added before applying CRC
mess = mess[:len(mess) - len(reminder)] + reminder
print("after replacing 0's by reminder : ", mess)

# reciever side CRC
reminder_new = divisionEx_OR(divisor, mess)
print("reminder after X-OR (reciever) : ", reminder_new)
if sum(int(x) for x in reminder_new) == 0:
    print("There is no error.")
else:
    print("There is some error.")

# # error
# 11001
# 1010101010
#
# # no any error
# 10011
# 1101011111
