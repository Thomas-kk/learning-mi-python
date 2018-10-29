import datetime
import sys

print("What is your name?")
name = input()
print('What is your age {}?'.format(name))
ageStr = input()
try:
    age = int(ageStr)
except ValueError:
    print("You did not enter a number")
    sys.exit(0)
print('You will be 100 years old in {}'.format(
    datetime.datetime.now().year-age+100))
print("Would you like to have the last massage printed again? (yes/no)")
con = input()
if(con.upper() == "YES" or con.upper() == "Y"):
    print("How many times?")
    n = int(input())
    for i in range(0, n-1):
        print('{}. You will be 100 years old in {}'.format(
            i, datetime.datetime.now().year-age+100))
