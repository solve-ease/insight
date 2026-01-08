import asyncpg
import logging
import os
import asyncio
from decimal import Decimal
from datetime import datetime, timezone
from server.services.vectorDB.main import delete_points, vector_db_add

logger = logging.getLogger(__name__)

class fileDB():
    def __init__(self):
        try:
            self.FILE_DB_TABLE_NAME = os.getenv("FILE_DB_TEABLE_NAME" , "files")
            logger.info("Connecting to file database")            

        except Exception as e:
            logger.error(f"error while connecting to the file database {e}")
    
    async def connect(self):
        self.conn = await asyncpg.connect(
                user="filedb_user",
                password="filedb_pass123",
                database="filedb",
                host="localhost"
            )
        
    async def check_files(self , files: list, rescan_start_time) -> list:
        # try:
        out = []

        for idx , i in enumerate(files):
            path = i[0]
            # logger.info(path)
            result = await self.conn.fetch("SELECT mtime, size FROM files WHERE path=$1",path)

            # logger.info(type(i[2]))

            if len(result) > 1:
                raise RuntimeError(f"File Database contains conflicting entries with the same path \n\n{result}")

            if not result:
                # logger.info("changed due to new file")
                out.append(path) # new or rename or move
            
            elif result[0][0] != i[1] or result[0][1] != i[2]: 
                # logger.info("chagned due to updated file")
                out.append(path) # changed
                await self.conn.fetch(f"UPDATE {self.FILE_DB_TABLE_NAME} SET rescan_time=$1 WHERE path=$2", datetime.now(timezone.utc), path)
            
            else:
                await self.conn.fetch(f"UPDATE {self.FILE_DB_TABLE_NAME} SET rescan_time=$1 WHERE path=$2", datetime.now(timezone.utc), path)
        
        # removing the files which are no longer present in the file system from the file database and the vector database
        points_delete = await self.conn.fetch(f"SELECT point_id FROM {self.FILE_DB_TABLE_NAME} WHERE rescan_time < $1", rescan_start_time)

        #implement in vector db service
        delete_points(points_delete)

        await self.conn.fetch(f"DELETE FROM {self.FILE_DB_TABLE_NAME} WHERE rescan_time < $1",rescan_start_time)
        return out
    
        # except Exception as e:
        #     logger.error(f"error while checking files {e}")
        #     raise
    
    async def check_file_hash(self , files : list):        
        for path , hash in files:
            query_results = await self.conn.fetch(f"SELECT path,hash FROM {self.FILE_DB_TABLE_NAME} WHERE hash=$1" , hash)

            if len(query_results) > 1:
                raise RuntimeError("Mulitple entries with the same hash value")
            
            stat = os.stat(path)
            mtime = stat.st_mtime
            size = stat.st_size
            
            rescan_time = datetime.now(timezone.utc)

            logger.info(path)

            if not query_results:
                # this means that this file is new and no other file is present.
                # edge case : there might be some possibility where the file is moved and changed together
                
                
                point_id = vector_db_add(path) # a method to add a media file in the vector db

                logger.info("new file detected")
                if point_id is None:
                    raise RuntimeError("Error while uploading media file in vector database")

                await self.conn.fetch(f"INSERT INTO {self.FILE_DB_TABLE_NAME} (hash, point_id , mtime, size, rescan_time, path) VALUES ($1,$2,$3,$4,$5,$6)", str(hash), str(point_id), mtime, size, rescan_time, path)
            
            else:
                #this means that the file is not new or changed and only the path has been changed.
                # so here we only need to update the respective entry in the files database and not create a new one
                # here one issue is that if a file is duplicate then the hashes would match but the paths woudl be different because names woudl be different, thus it may make some issues.

                logger.info("the path of the file was changed")

                await self.conn.fetch(f"UPDATE {self.FILE_DB_TABLE_NAME} SET rescan_time=$1, mtime=$2, size=$3, path=$4 WHERE hash=$5",rescan_time, mtime, size, path, hash)
        
        # except Exception as e:
        #     logger.error(f"Error occured while checking files hash: {e}")


    async def __del__ (self):
        try:
            logger.info("Closing file database connection")
            await self.conn.close()
        
        except Exception as e:
            logger.error(f"Error while closing connection with file database {e}")