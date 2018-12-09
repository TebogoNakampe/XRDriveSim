def calculate(calStr):
    try:
        calResult = eval(calStr)
    except:
        calResult = 0
        
    return str(calResult)