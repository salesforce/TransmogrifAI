/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
    val newf1 = f1.withMetadata(testMeta)
    
    val newReader = new CustomReader[Row](ReaderKey.randomKey) {
      def readFn(params: OpParams)(implicit spark: SparkSession): Either[RDD[Row], Dataset[Row]] = Right(ds)
    }
    val dataRead = newReader.generateDataFrame(Array(newf1, f2, f3), new OpParams())(spark)
    dataRead.drop(DataFrameFieldNames.KeyFieldName).collect() should contain theSameElementsAs ds.collect()
    dataRead.schema.filter(_.name == newf1.name).head.metadata shouldEqual testMeta
  }
}
