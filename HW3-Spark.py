# HW3 - Spark Assignment
# BDA 696 -  Karenina Zaballa

import sys

from pyspark import StorageLevel
from pyspark.sql import SparkSession

from Transformer_100_Day_AVG import R_AVG_100

# Setup Spark

# PERSONAL ERRORS I HAD TO RESEARCH ON:
# from setuptools import command
# process = subprocess.Popen(command, stdout=tempFile, shell=True)
# https://docs.python.org/3/library/subprocess.html


def main():
    spark = SparkSession.builder.master("local[*]").getOrCreate()

    database = "baseball_db"  # This may be named differently, i.e. "baseball"
    port = "3306"
    # input login info here
    user = " "  # pragma: allowlist secret
    password = " "  # pragma: allowlist secret

    # STANDARD TABLE
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
    # Make sure the big table shows
    df.show()

    df.createOrReplaceTempView("batters_counts")
    df.persist(StorageLevel.DISK_ONLY)

    df1 = (
        spark.read.format("jdbc")
        .options(
            url=f"jdbc:mysql://localhost:{port}/{database}",
            driver="com.mysql.cj.jdbc.Driver",
            dbtable="(SELECT game.game_id, batter_counts.batter, \
                    batter_counts.Hit, batter_counts.atBat, game.local_date \
                    FROM batter_counts\
                    JOIN game ON batter_counts.game_id = game.game_id)ba_temp",
            user=user,  # pragma: allowlist secret
            password=password,  # pragma: allowlist secret
        )
        .load()
    )
    # Make sure ba_temp shows
    df1.show()

    df1.createOrReplaceTempView("ba_temp")
    df1.persist(StorageLevel.DISK_ONLY)

    df1.createOrReplaceTempView("ba_temp")

    # Paste ba_temp onto itself to set up for calculations
    tableprep = spark.sql(
        """SELECT ba1.batter, ba1.game_id, SUM(ba2.Hit) AS Hit_Sum, \
            SUM(ba2.atBat) AS AtBat_Sum, ba2.local_date \
            FROM   ba_temp ba1 \
            JOIN   ba_temp ba2 \
            ON ba1.batter = ba2.batter \
            AND ba2.local_date > DATE_SUB(ba1.local_date, INTERVAL 100 DAY) \
            AND ba1.local_date > ba2.local_date \
            GROUP BY ba1.game_id, ba1.batter,ba1.local_date """
    )

    # set up your input columns for calculation
    roll_avg_100 = R_AVG_100(
        inputCols=["Hit_Sum", "AtBat_Sum"], outputCol="Rolling_Bat_AVG_100"
    )
    # this step calls your function from
    # Transformer_100_Day_AVG.py and does your calculation
    rolling = roll_avg_100.transform(tableprep)
    print("Showing 100 Day Rolling Average for each batter\n")
    rolling.show()


if __name__ == "__main__":
    sys.exit(main())
