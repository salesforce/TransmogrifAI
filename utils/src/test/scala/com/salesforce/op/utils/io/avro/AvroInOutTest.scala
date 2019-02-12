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

package com.salesforce.op.utils.io.avro

import java.io.{File, FileNotFoundException, FileWriter}
import java.nio.file.Paths

import com.salesforce.op.test.TestSparkContext
import com.salesforce.op.utils.io.avro.AvroInOut._
import org.apache.avro.generic.GenericRecord
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.rdd.RDD
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class AvroInOutTest extends FlatSpec with TestSparkContext {
  val avroSchemaPath = s"$testDataDir/PassengerDataAll.avsc"
  val avroFilePath = s"$testDataDir/PassengerDataAll.avro"
  val avroFileRecordCount = 891
  val hdfs: FileSystem = FileSystem.get(sc.hadoopConfiguration)
  lazy val avroTemp: String = tempDir + "/avro-inout-test"

  Spec(AvroInOut.getClass) should "creates RDD from an avro file" in {
    val res = readPathSeq(avroFilePath, withCount = true, deepCopy = true, persist = false)
    res shouldBe a[RDD[_]]
    res.count shouldBe avroFileRecordCount
  }

  it should "creates RDD from a sequence of avro files" in {
    val res = readPathSeq(s"$avroFilePath,$avroFilePath")
    res.count shouldBe avroFileRecordCount*2
  }

  it should "create RDD from a mixed sequence of valid and invalid avro files" in {
    val res = readPathSeq(s"badfile/path1,$avroFilePath,badfile/path2,$avroFilePath,badfile/path3")
    res.count shouldBe avroFileRecordCount*2
  }

  it should "throw an error if passed in avro files are invalid" in {
    val error = intercept[IllegalArgumentException](readPathSeq("badfile/path1,badfile/path2"))
    error.getMessage shouldBe "No valid directory found in path 'badfile/path1,badfile/path2'"
  }

  it should "creates Some(RDD) from an avro file" in {
    val res = read(avroFilePath)
    res.size shouldBe 1
    res.get shouldBe an[RDD[_]]
    res.get.count shouldBe avroFileRecordCount
  }

  it should "create None from an invalid avro file" in {
    val res = read("badfile/path")
    res shouldBe None
  }

  Spec[AvroWriter[_]] should "writeAvro to filesystem" in {
    val avroData = readPathSeq(avroFilePath).asInstanceOf[RDD[GenericRecord]]
    val avroSchema = loadFile(avroSchemaPath)

    val error = intercept[FileNotFoundException](hdfs.listStatus(new Path(avroTemp)))
    error.getMessage shouldBe s"File $avroTemp does not exist"

    AvroWriter(avroData).writeAvro(avroTemp, avroSchema)
    val hdfsFiles = hdfs.listStatus(new Path(avroTemp)) filter (x => x.getPath.getName.contains("part"))
    val res = readPathSeq((for { x <- hdfsFiles } yield avroTemp + "/" + x.getPath.getName).mkString(","))
    res.count shouldBe avroFileRecordCount
  }

  it should "checkPathsExist" in {
    val tmpDir = Paths.get(File.separator, "tmp").toFile
    val f1 = new File(tmpDir, "avroinouttest")
    f1.delete()
    val w = new FileWriter(f1)
    w.write("just checking")
    w.close()
    val f2 = new File(tmpDir, "thisfilecannotexist")
    f2.delete()
    val f3 = new File(tmpDir, "this file cannot exist")
    f3.delete()
    assume(f1.exists && !f2.exists && !f3.exists)

    selectExistingPaths(s"$f1,$f2") shouldBe f1.toString

    intercept[IllegalArgumentException] { selectExistingPaths(f2.toString) }
    intercept[IllegalArgumentException] { selectExistingPaths(f3.toString) }

    selectExistingPaths(s"$f1,$f3") shouldBe f1.toString

    intercept[IllegalArgumentException] { selectExistingPaths("") }
  }

}
