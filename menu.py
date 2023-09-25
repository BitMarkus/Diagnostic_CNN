#######################################
# Functions for validating user input #
#######################################

# Creates an input with prompt
# which is checked, if the input is an integer number
# If not, the loop will continue until a valid number is entered
def input_int(prompt):
    while(True):
        nr = input(prompt)
        if not(check_int(nr)):
            print("Input is not an integer number! Try again:")
        else:
            return int(nr)   

# Creates an input with prompt
# which is checked, if the volume is in a valid format
# If not, the loop will continue until a valid string is entered
# Or the exit string is entered
# Cannot be negative and needs to be an int or float
def input_volume(prompt):
    while(True):
        nr = input(prompt)
        if(exit_menu(nr)):
            return False
        elif not(check_nr(nr)):
            print("Input is not a number! Try again:")
        else:
            if(float(nr) <= 0):
                print("Volume cannot be negative! Try again:")
            else:
                # Check if volume has more than three digits after dot
                if(get_dec(str(nr)) > 3):
                    print("MiC can only have maximum three decimals! Try again:")
                else:
                    return float(nr)    

####################################        
# Functions to check variable type #
####################################

# Check variable for number (int or float)
# Returns True if conversion was successful
# or False when the variable cannot be converted to a number
def check_nr(var):
    try:
        # Convert it into integer
        val = int(var)
        return True
    except ValueError:
        try:
            # Convert it into float
            val = float(var)
            return True
        except ValueError:
            return False

# Check variable for int
# Returns True if conversion was successful
# or False when the variable cannot be converted to an integer number
def check_int(var):
    try:
        val = int(var)
        return True
    except ValueError:
        return False
    
# Check variable for float
# Returns the variable as int if conversion was successful
# or False when the variable cannot be converted to an integer number
def check_float(var):
    try:
        # Convert it into integer
        val = int(var)
        return False
    except ValueError:
        try:
            # Convert it into float
            val = float(var)
            return True
        except ValueError:
            return False

##################
# Misc functions #
##################

# Function to exit any menue
def exit_menu(var, stop = "<exit>"):
    if(var == stop):
        print("Input canceled!")
        return True
    else:
        return False
    
# Prints a message to exit a menue with <exit>
def exit_menu_msg(stop = "<exit>"):
    print(f"> Enter {stop} to return to menue")
    
# Function returns number of decimal places after decimal point
# source: https://stackoverflow.com/questions/28749177/how-to-get-number-of-decimal-places
def get_dec(no_str):
    if "." in no_str:
        return len(no_str.split(".")[1].rstrip("0"))
    else:
        return 0