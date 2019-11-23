from azure.cosmosdb.table.tableservice import TableService
from azure.cosmosdb.table.models import Entity
from datetime import datetime

import logging
logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("TableService").setLevel(logging.WARNING)


storage_name="textae"
key=r"6yBCXlblof8DVFJ4BD3eNFTrGQCej6cKfCf5z308cKnevyHaG+yl/m+ITVErB9yt0kvN3ToqxLIh0knJEfFmPA=="
ts = TableService(account_name=storage_name, account_key=key)

# ts.create_table('firsttable')
table_name = 'firsttable'

logger.info("Insert row into Table %s", table_name)

row = {
        'PartitionKey': 'MILU_Rule_Rule_Template',
        'RowKey': str(datetime.now()),
        'iter': str(1)
    }

ts.insert_entity(table_name, row)