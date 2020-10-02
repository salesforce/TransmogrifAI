package com.salesforce.op.utils.io.file

import java.io.{BufferedInputStream, FileInputStream, FileOutputStream}
import java.nio.file.{Files, Paths}
import java.util.zip.{ZipEntry, ZipFile, ZipOutputStream}

import org.apache.commons.io.FileUtils

import scala.collection.JavaConverters._

case object FileCompress {

  /**
   * Zip a directory
   * @param inLoc location of directory to zip
   * @param zipLoc output zip file name
   */
  def zip(inLoc: String, zipLoc: String): Unit = {
    val inPath = Paths.get(inLoc)
    val zipPath = Paths.get(zipLoc)
    val allFiles = FileUtils.listFiles(inPath.toFile, null, true)
    val zipFile = new ZipOutputStream(new FileOutputStream(zipPath.toString))
    for {entry <- allFiles.asScala} {
      val name = entry.getPath.stripPrefix("file:")
      zipFile.putNextEntry(new ZipEntry(name))
      val in = new BufferedInputStream(new FileInputStream(name))
      var b = in.read()
      while (b > -1) {
        zipFile.write(b)
        b = in.read()
      }
      in.close()
      zipFile.closeEntry()
    }
    zipFile.close()
    FileUtils.deleteDirectory(inPath.toFile)
  }

  /**
   * Unzip a directory
   * @param zipLoc zipped file location
   * @param outLoc output directory location
   */
  def unzip(zipLoc: String, outLoc: String): Unit = {
    val zipPath = Paths.get(zipLoc)
    val outPath = Paths.get(outLoc)
    val zipFile = new ZipFile(zipPath.toFile)
    for {entry <- zipFile.entries.asScala} {
      val path = outPath.resolve(entry.toString)
      if (entry.isDirectory) {
        Files.createDirectories(path)
      } else {
        Files.createDirectories(path.getParent)
        Files.copy(zipFile.getInputStream(entry), path)
      }
    }
    FileUtils.deleteDirectory(zipPath.toFile)
  }

}
