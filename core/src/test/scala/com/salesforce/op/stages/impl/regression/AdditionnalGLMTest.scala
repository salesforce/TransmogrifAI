package com.salesforce.op.stages.impl.regression

import com.salesforce.op.OpWorkflow
import com.salesforce.op.evaluators.Evaluators
import com.salesforce.op.features.{Feature, FeatureLike}
import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.selector.ModelSelectorSummary
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.testkit.{RandomReal, RandomVector}
import org.apache.spark.ml.linalg._
import org.scalatest.FlatSpec
import com.salesforce.op.utils.spark.RichMetadata._
import org.junit.runner.RunWith
import org.scalacheck.Gen
import org.scalatest.junit.JUnitRunner
import org.scalatest.prop.Checkers

@RunWith(classOf[JUnitRunner])
class AdditionnalGLMTest extends FlatSpec with TestSparkContext with Checkers {

  import org.scalacheck.Prop

  val rowInteger = Gen.choose(500, 1000)
  val columnInteger = Gen.choose(10, 50)


  "Regression Model Selector" should "Not pick Linear Regression when the response are counts" in {
    check(Prop.forAll(rowInteger) { n =>
      Prop.forAll(columnInteger) { p =>
        Prop.collect(n, p) {
          println(s"n = $n p = $p")
          //      val (data, features) = TestFeatureBuilder.random(n)()
          //      data.show()
          //      println(data.schema.fields.toSeq.map(_.dataType))
          //      val (dep, indep) = features.toSeq.partition(_.typeName.endsWith("Integral"))
          //      println(indep.map(f => f.name -> f.typeName))
          //
          //      val response = dep.head.asInstanceOf[FeatureLike[Integral]].map(_.toDouble.map(math.abs).toRealNN(0.0))
          //        .asInstanceOf[Feature[RealNN]].copy(isResponse = true)
          //      val featureVector = indep.filter(f => !(f.typeName.contains("Text"))).transmogrify()
          //      val newData = new OpWorkflow().setResultFeatures(response, featureVector).transform(data)
          //      newData

          val vectors = RandomVector.dense(RandomReal.exponential(mean = 1.0), p).take(n).toSeq
          val mean: Double = spark.sparkContext.parallelize(vectors.map(_.value.toArray.sum)).mean()

          println("Yup")
          val labels = //vectors.map(vector => math.floor(math.exp(vector.value.toArray.sum)).toRealNN)
            RandomReal.poisson[RealNN](mean = mean).limit(n).toSeq


          println("Done")
          val (data, features, label) = TestFeatureBuilder("features", "response", vectors.zip(labels))
          val response = label.copy(isResponse = true)

          data.show
          val selector = RegressionModelSelector.withCrossValidation(modelTypesToUse =
            Seq(RegressionModelsToTry.OpGeneralizedLinearRegression,RegressionModelsToTry.OpRandomForestRegressor,
              RegressionModelsToTry.OpLinearRegression),
            numFolds = 5
           // validationMetric = Evaluators.Regression.r2()
           ).setInput(response, features)
          val model = selector.fit(data)
          println(model.getSparkMlStage())
          println(ModelSelectorSummary.fromMetadata(model.getMetadata().getSummaryMetadata).validationResults
            .map(v => v.modelName -> v.metricValues).sortBy(- _._2.toMap.values.head.asInstanceOf[Double]))
          model.transform(data)
          !model.getSparkMlStage().get.toString().contains("linReg")
          //data.count() == n
        }
      }
    })

    //check(Prop.forAll { (m: Int, n: Int) => Prop.collect(m, n, m+n) { m + n != 37 } })
  }

}
