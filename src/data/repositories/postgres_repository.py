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