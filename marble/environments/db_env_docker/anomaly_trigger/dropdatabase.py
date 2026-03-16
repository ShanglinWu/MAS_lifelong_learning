import psycopg2
from utils.database import DB_CONFIG


def dropdatabase(name):
    # 连接到 "postgres" 数据库
    conn = psycopg2.connect(
        dbname=DB_CONFIG["dbname"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
    )

    conn.autocommit = True
    # 创建一个数据库游标
    cur = conn.cursor()

    # 删除数据库的 SQL 语句
    drop_database_query = f"DROP DATABASE IF EXISTS {name}"

    # 执行删除数据库的 SQL 语句
    cur.execute(drop_database_query)

    # 提交更改
    conn.commit()

    # 关闭游标和数据库连接
    cur.close()
    conn.close()
