import logging
from datetime import datetime
import os

FILE_PATH = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
log_path = os.path.join(os.getcwd(),'logs')
os.makedirs(log_path, exist_ok=True)
final_path = os.path.join(log_path, FILE_PATH)

logging.basicConfig(
    filename=final_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


logging.info("Logger initialized")
logging.info(f"Log file created at: {final_path}")