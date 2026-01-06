import asyncpg
import logging

logger = logging.getLogger(__name__)

class fileDB():
    async def __init__(self):
        try:
            logger.info("Connecting to file database")
            
            self.conn = await asyncpg.connect(
                user="myuser",
                password="mypassword",
                database="mydb",
                host="localhost"
            )

        except Exception as e:
            logger.error(f"error while connecting to the file database {e}")

    async def check_files(self , files: list) -> list:
        try:
            out = []

            for idx , i in enumerate(files):
                path = i[0]
                result = await self.conn.fetch(f"SELECT mtime, size FROM files WHERE path={path}")

                if len(result) > 1:
                    raise RuntimeError(f"File Database contains conflicting entries with the same path \n\n{result}")

                if not result:
                    out[idx] = path # new or rename or move
                
                elif result[0][0] != i[1] or result[0][0] != i[2]: 
                    out[idx] = path # changed
                
            return out
        
        except Exception as e:
            logger.error(f"error while checking files {e}")
            raise
    
    async def __del__ (self):
        try:
            logger.info("Closing file database connection")
            await self.conn.close()
        
        except Exception as e:
            logger.error(f"Error while closing connection with file database {e}")