/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.test

import java.io.{File, IOException}
import java.util.UUID

import org.scalatest.{BeforeAndAfterAll, Suite}

/**
 * Trait that creates a temporary directory before all tests and deletes it after all.
 * Copied from Spark test suite
 */
trait TempDirectoryTest extends BeforeAndAfterAll { self: Suite =>

  private var _tempDir: File = _

  /** Returns the temporary directory as a [[File]] instance. */
  protected def tempDir: File = _tempDir

  private val shutdownDeletePaths = new scala.collection.mutable.HashSet[String]()

  /**
   * Create a directory inside the given parent directory. The directory is guaranteed to be
   * newly created, and is not marked for automatic deletion.
   */
  def createDirectory(root: String, namePrefix: String = "spark"): File = {
    var attempts = 0
    val maxAttempts = 10
    var dir: File = null
    while (dir == null) {
      attempts += 1
      if (attempts > maxAttempts) {
        throw new IOException("Failed to create a temp directory (under " + root + ") after " +
          maxAttempts + " attempts!")
      }
      try {
        dir = new File(root, namePrefix + "-" + UUID.randomUUID.toString)
        if (dir.exists() || !dir.mkdirs()) {
          dir = null
        }
      } catch {
        case e: SecurityException => dir = null;
      }
    }

    dir.getCanonicalFile
  }

  // Register the path to be deleted via shutdown hook
  def registerShutdownDeleteDir(file: File): Unit = {
    val absolutePath = file.getAbsolutePath()
    shutdownDeletePaths.synchronized {
      shutdownDeletePaths += absolutePath
    }
  }

  /**
   * Create a temporary directory inside the given parent directory. The directory will be
   * automatically deleted when the VM shuts down.
   */
  def createTempDir(
    root: String = System.getProperty("java.io.tmpdir"),
    namePrefix: String = "spark"): File = {
    val dir = createDirectory(root, namePrefix)
    registerShutdownDeleteDir(dir)
    dir
  }

  // Remove the path to be deleted via shutdown hook
  def removeShutdownDeleteDir(file: File): Unit = {
    val absolutePath = file.getAbsolutePath()
    shutdownDeletePaths.synchronized {
      shutdownDeletePaths.remove(absolutePath)
    }
  }

  /**
   * Check to see if file is a symbolic link.
   */
  def isSymlink(file: File): Boolean = {
    if (file == null) throw new NullPointerException("File must not be null")
    val fileInCanonicalDir = if (file.getParent() == null) {
      file
    } else {
      new File(file.getParentFile().getCanonicalFile(), file.getName())
    }

    !fileInCanonicalDir.getCanonicalFile().equals(fileInCanonicalDir.getAbsoluteFile())
  }

  private def listFilesSafely(file: File): Seq[File] = {
    if (file.exists()) {
      val files = file.listFiles()
      if (files == null) {
        throw new IOException("Failed to list files for dir: " + file)
      }
      files
    } else {
      List()
    }
  }

  /**
   * Delete a file or directory and its contents recursively.
   * Don't follow directories if they are symlinks.
   * Throws an exception if deletion is unsuccessful.
   */
  def deleteRecursively(file: File): Unit = {
    if (file != null) {
      try {
        if (file.isDirectory && !isSymlink(file)) {
          var savedIOException: IOException = null
          for { child <- listFilesSafely(file) } {
            try {
              deleteRecursively(child)
            } catch {
              // In case of multiple exceptions, only last one will be thrown
              case ioe: IOException => savedIOException = ioe
            }
          }
          if (savedIOException != null) {
            throw savedIOException
          }
          removeShutdownDeleteDir(file)
        }
      } finally {
        if (!file.delete()) {
          // Delete can also fail if the file simply did not exist
          if (file.exists()) {
            throw new IOException("Failed to delete: " + file.getAbsolutePath)
          }
        }
      }
    }
  }

  override def beforeAll: Unit = {
    super.beforeAll()
    _tempDir = createTempDir(namePrefix = this.getClass.getName)
  }

  override def afterAll: Unit = {
    try {
      deleteRecursively(_tempDir)
    } finally {
      super.afterAll()
    }
  }
}
