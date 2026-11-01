import psycopg
from psycopg import sql
from collections.abc import Iterator
from psycopg.rows import dict_row


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

    def read_batches(
        self,
        schema: str,
        table: str,
        batch_size: int = 1000
    ) -> Iterator[list[dict]]:

        if batch_size <= 0:
            raise ValueError("batch_size debe ser mayor que cero.")

        query = sql.SQL(
            "SELECT * FROM {}"
        ).format(
            sql.Identifier(schema, table)
        )

        with self.connect() as connection:
            with connection.cursor(
                name="table_reader",
                row_factory=dict_row
            ) as cursor:
                cursor.execute(query)

                while True:
                    batch = cursor.fetchmany(batch_size)

                    if not batch:
                        break

                    yield batch
    
    
    def create_typed_table(
        self,
        schema: str,
        table: str,
        column_types: dict[str, str]
    ) -> None:

        if not column_types:
            raise ValueError("El mapa de tipos está vacío.")

        postgres_types = {
            "string": "TEXT",
            "numeric": "NUMERIC"
        }

        column_definitions = []

        for column, column_type in column_types.items():
            if column_type not in postgres_types:
                raise ValueError(
                    f"Tipo no soportado para {column}: {column_type}"
                )

            column_definitions.append(
                sql.SQL("{} {}").format(
                    sql.Identifier(column),
                    sql.SQL(postgres_types[column_type])
                )
            )

        query = sql.SQL(
            "CREATE TABLE IF NOT EXISTS {} ({})"
        ).format(
            sql.Identifier(schema, table),
            sql.SQL(", ").join(column_definitions)
        )

        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
    
    
    def count_rows(
        self,
        schema: str,
        table: str
    ) -> int:

        query = sql.SQL(
            "SELECT COUNT(*) FROM {}"
        ).format(
            sql.Identifier(schema, table)
        )

        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                return cursor.fetchone()[0]

    def get_table_columns(
        self,
        schema: str,
        table: str
    ) -> dict[str, str]:

        query = """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = %s
              AND table_name = %s
            ORDER BY ordinal_position
        """

        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (schema, table))
                return dict(cursor.fetchall())
    
    def read_by_ids(
        self,
        schema: str,
        table: str,
        identifiers: list[str],
        key_column: str = "Customer_ID"
    ) -> list[dict]:

        if not identifiers:
            return []

        query = sql.SQL(
            "SELECT * FROM {} WHERE {} = ANY(%s)"
        ).format(
            sql.Identifier(schema, table),
            sql.Identifier(key_column)
        )

        with self.connect() as connection:
            with connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(query, (identifiers,))
                return cursor.fetchall()