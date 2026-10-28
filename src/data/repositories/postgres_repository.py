import psycopg
from psycopg import sql


class PostgresRepository:

    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str
    ):
        self.connection_params = {
            "host": host,
            "port": port,
            "dbname": database,
            "user": user,
            "password": password
        }

    def connect(self):
        return psycopg.connect(
            **self.connection_params
        )
    def create_table(
        self,
        schema: str,
        table: str,
        columns: list[str]
    ) -> None:

        if not columns or any(not column.strip() for column in columns):
            raise ValueError("La lista de columnas no es válida.")

        if len(columns) != len(set(columns)):
            raise ValueError("Existen columnas duplicadas.")

        column_definitions = [
            sql.SQL("{} TEXT").format(
                sql.Identifier(column)
            )
            for column in columns
        ]

        query = sql.SQL(
            "CREATE TABLE IF NOT EXISTS {} ({})"
        ).format(
            sql.Identifier(schema, table),
            sql.SQL(", ").join(column_definitions)
        )

        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)

    
    def insert_batch(
        self,
        connection,
        schema: str,
        table: str,
        columns: list[str],
        rows: list[dict]
    ) -> int:

        if not rows:
            return 0

        query = sql.SQL(
            "INSERT INTO {} ({}) VALUES ({})"
        ).format(
            sql.Identifier(schema, table),
            sql.SQL(", ").join(
                sql.Identifier(column)
                for column in columns
            ),
            sql.SQL(", ").join(
                sql.Placeholder()
                for _ in columns
            )
        )

        values = [
            tuple(row[column] for column in columns)
            for row in rows
        ]

        with connection.cursor() as cursor:
            cursor.executemany(query, values)

        return len(rows)

    def truncate_table(
            self,
            connection,
            schema: str,
            table: str
        ) -> None:

            query = sql.SQL(
                "TRUNCATE TABLE {}"
            ).format(
                sql.Identifier(schema, table)
            )

            with connection.cursor() as cursor:
                cursor.execute(query)

    
    def create_schema(
        self,
        schema: str
    ) -> None:

        query = sql.SQL(
            "CREATE SCHEMA IF NOT EXISTS {}"
        ).format(
            sql.Identifier(schema)
        )

        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
