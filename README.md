# ShouldTire

Working for a bit as a automotive technician, the amount of tires coming in and out that were defective was significant.
Driving on defective tires pose not only a hazard to the driver but also the surroundings.
Since automotive shops are often overwhelmed and vehicle owners cannot discern reliable tires on their own,
this application is an experiment whether the monitoring of tire degradation could be automated.

## Installation

*Local Dev*
1. Make sure you have [Python 3.10 and PIP ↗](https://www.python.org/downloads/release/python-31020/).
2. Build python env
`python -m venv capstone-backend-env`
3. Install dependencies
`pip install -r requirements.txt`
4. Activate env
```
ShouldTire_ENV\Scripts\activate.bat // cmd
source ShouldTire_ENV/Scripts/activate // bash
ShouldTire_ENV\Scripts\Activate.ps1 // powershell
```
5. Build app 
`python main.py`
6. Server is ready
`http://127.0.0.1:8050/`

## Stack
Frontend -> Dash<br>
Backend -> Python<br>
