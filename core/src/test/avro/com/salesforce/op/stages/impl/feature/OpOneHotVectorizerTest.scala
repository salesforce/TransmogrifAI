package com.salesforce.op.stages.impl.feature

import com.salesforce.op.features.{FeatureBuilder, FeatureSparkTypes}
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.features.types._
import com.salesforce.op.testkit.{RandomReal, RandomVector}
import com.salesforce.op.utils.reflection.ReflectionUtils
import com.salesforce.op.utils.spark.SequenceAggregators
import com.twitter.algebird.HLL
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.junit.runner.RunWith
import org.scalatest.{FlatSpec, FunSuite}
import org.scalatest.junit.JUnitRunner

import scala.reflect.ClassTag

@RunWith(classOf[JUnitRunner])
class OpOneHotVectorizerTest extends FlatSpec with TestSparkContext with OneHotFun {

  val bits = 12
  implicit val classTag: ClassTag[Double] = ReflectionUtils.classTagForWeakTypeTag[Double]
  implicit val kryo = new KryoSerializer(spark.sparkContext.getConf)

  lazy val bigData = createData(10000, 1000)

  lazy val smallData = createData(100, 10)

  lazy val bigRows = createData(10000, 10)

  lazy val bigCols = createData(100, 1000)

  it should "Compare for Big Data" in {
    val m = bigData.columns.size
    val t0 = System.nanoTime()
    val hll = SequenceAggregators.HLLSeq[Double](size = m, bits = bits)
    val uniqueCounts1 = bigData.select(hll.toColumn).first()
    val t1 = System.nanoTime()
    val uniqueCounts2 = countUniques(bigData, size = m, bits = bits)
    val t2 = System.nanoTime()

    println("Elapsed time for Proposal 1: " + (t1 - t0) * 1e-9 + "s")
    println("Elapsed time for Proposal 2: " + (t2 - t1) * 1e-9 + "s")

  }

  it should "Compare for Small Data" in {
    val m = smallData.columns.size
    val t0 = System.nanoTime()
    val hll = SequenceAggregators.HLLSeq[Double](size = m, bits = bits)
    val uniqueCounts1 = smallData.select(hll.toColumn).first()
    val t1 = System.nanoTime()
    val uniqueCounts2 = countUniques(smallData, size = m, bits = bits)
    val t2 = System.nanoTime()

    println("Elapsed time for Proposal 1: " + (t1 - t0) * 1e-9 + "s")
    println("Elapsed time for Proposal 2: " + (t2 - t1) * 1e-9 + "s")

  }

  it should "Compare for lots of Rows" in {
    val m = bigRows.columns.size
    val t0 = System.nanoTime()
    val hll = SequenceAggregators.HLLSeq[Double](size = m, bits = bits)
    val uniqueCounts1 = bigRows.select(hll.toColumn).first()
    val t1 = System.nanoTime()
    val uniqueCounts2 = countUniques(bigRows, size = m, bits = bits)
    val t2 = System.nanoTime()

    println("Elapsed time for Proposal 1: " + (t1 - t0) * 1e-9 + "s")
    println("Elapsed time for Proposal 2: " + (t2 - t1) * 1e-9 + "s")

  }

  it should "Compare for lots of Columns" in {
    val m = bigCols.columns.size
    val t0 = System.nanoTime()
    val hll = SequenceAggregators.HLLSeq[Double](size = m, bits = bits)
    val uniqueCounts1 = bigCols.select(hll.toColumn).first()
    val t1 = System.nanoTime()
    val uniqueCounts2 = countUniques(bigCols, size = m, bits = bits)
    val t2 = System.nanoTime()

    println("Elapsed time for Proposal 1: " + (t1 - t0) * 1e-9 + "s")
    println("Elapsed time for Proposal 2: " + (t2 - t1) * 1e-9 + "s")

  }



  def createData(nRows: Int, nCols:Int): Dataset[Seq[Double]] = {
    val fv = RandomVector.dense(RandomReal.poisson(10), nCols).limit(nRows)

    val seq = fv.map(_.value.toArray.toSeq.map(_.toReal))
    val features = (0 until nCols).map(i => FeatureBuilder.fromRow[Real](i.toString).asPredictor)

    val schema = FeatureSparkTypes.toStructType(features: _ *)

    implicit val rowEncoder = RowEncoder(schema)

    import spark.implicits._

    val data = seq.map(p => Row.fromSeq(
      p.map { case f: FeatureType => FeatureTypeSparkConverter.toSpark(f) }
    )).toDF().map(_.toSeq.map(_.asInstanceOf[Double]))

    data.persist
  }
}
