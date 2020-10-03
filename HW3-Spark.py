# Setup Spark


from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local[*]").getOrCreate()

database = "baseball_db"
port = "3306"
user = "root"  # pragma: allowlist secret
password = "x11docker"  # pragma: allowlist secret

df = (
    spark.read.format("jdbc")
    .options(
        url=f"jdbc:mysql://localhost:{port}/{database}",
        driver="com.mysql.cj.jdbc.Driver",
        dbtable="batter_counts",
        user=user,  # pragma: allowlist secret
        password=password,  # pragma: allowlist secret
    )
    .load()
)

df.show()
