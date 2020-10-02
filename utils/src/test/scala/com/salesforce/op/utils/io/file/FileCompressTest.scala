package com.salesforce.op.utils.io.file

import java.nio.file.Paths

import com.salesforce.op.test.{TempDirectoryTest, TestCommon}
import org.apache.commons.io.FileUtils
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

import scala.collection.JavaConverters._

@RunWith(classOf[JUnitRunner])
class FileCompressTest extends FlatSpec with TempDirectoryTest with TestCommon{

  private val localDir = resourceFile(name = "zipTestFiles/rawModel")
  private val localZipped = resourceFile(name = "zipTestFiles.Model.zip")



  Spec(FileCompress.getClass) should "compress a directory into a zip file" in {
    val zipFile = tempDir + "/Model.zip"
    FileCompress.zip(localDir.getPath, zipFile)
    Paths.get(zipFile).toFile.exists() shouldBe true
  }

  it should "uncompress a zip file into a directory" in {
    val unzipFile = tempDir + "/ModelUnzipped"
    FileCompress.unzip(localZipped.getPath, unzipFile)
    Paths.get(unzipFile).toFile.isDirectory shouldBe true
  }

  it should "round trip a file and get the same values" in {
    val zipFile = tempDir + "/Model.zip"
    val unzipFile = tempDir + "/ModelUnzipped"
    FileCompress.zip(localDir.getPath, zipFile)
    FileCompress.unzip(zipFile, unzipFile)
    val originalFiles = FileUtils.listFiles(localDir, null, true)
    val unzippedFiles = FileUtils.listFiles(Paths.get(unzipFile).toFile, null, true)
    originalFiles.asScala.map(_.toString) should contain theSameElementsAs unzippedFiles.asScala.map(_.toString)
  }

}
