/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op._
import com.salesforce.op.features.types._
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.ml.feature.IDF
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.ml.{Estimator, Transformer}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Assertions, FlatSpec, Matchers}


@RunWith(classOf[JUnitRunner])
class IDFTest extends FlatSpec with TestSparkContext {

  val data = Seq(
    Vectors.sparse(4, Array(1, 3), Array(1.0, 2.0)),
    Vectors.dense(0.0, 1.0, 2.0, 3.0),
    Vectors.sparse(4, Array(1), Array(1.0))
  )

  lazy val (ds, f1) = TestFeatureBuilder(data.map(_.toOPVector))

  Spec[IDF] should "compute inverted document frequency" in {
    val idf = f1.idf()
    val model = idf.originStage.asInstanceOf[Estimator[_]].fit(ds)
    val transformedData = model.asInstanceOf[Transformer].transform(ds)
    val results = transformedData.select(idf.name).collect(idf)

    idf.name shouldBe idf.originStage.outputName

    val expectedIdf = Vectors.dense(Array(0, 3, 1, 2).map { x =>
      math.log((data.length + 1.0) / (x + 1.0))
    })
    val expected = scaleDataWithIDF(data, expectedIdf)

    for {
      (res, exp) <- results.zip(expected)
      (x, y) <- res.value.toArray.zip(exp.toArray)
    } assert(math.abs(x - y) <= 1e-5)
  }

  it should "compute inverted document frequency when minDocFreq is 1" in {
    val idf = f1.idf(minDocFreq = 1)
    val model = idf.originStage.asInstanceOf[Estimator[_]].fit(ds)
    val transformedData = model.asInstanceOf[Transformer].transform(ds)
    val results = transformedData.select(idf.name).collect(idf)
    idf.name shouldBe idf.originStage.outputName

    val expectedIdf = Vectors.dense(Array(0, 3, 1, 2).map { x =>
      if (x > 0) math.log((data.length + 1.0) / (x + 1.0)) else 0
    })
    val expected = scaleDataWithIDF(data, expectedIdf)

    for {
      (res, exp) <- results.zip(expected)
      (x, y) <- res.value.toArray.zip(exp.toArray)
    } assert(math.abs(x - y) <= 1e-5)
  }

  private def scaleDataWithIDF(dataSet: Seq[Vector], model: Vector): Seq[Vector] = {
    dataSet.map {
      case data: DenseVector =>
        val res = data.toArray.zip(model.toArray).map { case (x, y) => x * y }
        Vectors.dense(res)
      case data: SparseVector =>
        val res = data.indices.zip(data.values).map { case (id, value) =>
          (id, value * model(id))
        }
        Vectors.sparse(data.size, res)
    }
  }

}
