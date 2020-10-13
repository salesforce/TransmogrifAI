package com.salesforce.op.readers

import com.salesforce.op.OpParams
import com.salesforce.op.test.{TestCommon, TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.RichDataset
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.MetadataBuilder
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner
import com.salesforce.op.features.types._

@RunWith(classOf[JUnitRunner])
class CustomReaderTest extends FlatSpec with TestCommon with TestSparkContext {

  Spec[CustomReader[_]] should "work with a dataframe and create metadata when added to the features" in {
    val (ds, f1, f2, f3) = TestFeatureBuilder(Seq(
      (1.0.toReal, 2.0.toReal, 3.0.toReal),
      (1.0.toReal, 2.0.toReal, 3.0.toReal),
      (1.0.toReal, 2.0.toReal, 3.0.toReal)
    ))
    val testMeta = new MetadataBuilder().putString("test", "myValue").build()
    val newf1 = f1.copy(name = "test", metadata = Option(testMeta))


    val newReader = new CustomReader[Row](ReaderKey.randomKey) {
      def readFn(params: OpParams)(implicit spark: SparkSession): Either[RDD[Row], Dataset[Row]] = Right(ds)
    }
    val dataRead = newReader.generateDataFrame(Array(newf1, f2, f3), new OpParams())(spark)
    dataRead.drop(DataFrameFieldNames.KeyFieldName).collect() should contain theSameElementsAs ds.collect()
    dataRead.schema.filter(_.name == newf1.name).head.metadata shouldEqual testMeta
  }
}
