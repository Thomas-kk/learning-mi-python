import sys

def oddOrEven():
    print("Please input a number")
    numStr = input()
    try:
        num = int(numStr)
    except ValueError:
        print("You did not enter a number, try again")
        oddOrEven()
        sys.exit(0)
    if(num % 4 == 0):
        print("Number {} is even and a multiple of 4".format(num))
    elif(num % 2 == 0):
        print("Number {} is even".format(num))
    else:
        print("Number {} is odd".format(num))

oddOrEven()