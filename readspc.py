"""Module to read SPC files.

Provides a read() function to read and return the entirety of an spcfile.

    hdr, subhdr, x, y = read(spcfile, **kwargs)

Or also provides an object oriented interface through the spcreader object.

    reader = spcreader(filename)
    reader.read()

Now spcreader behaves as an iterable to yield each spectrum in the file.

"""

import os
import struct
import warnings
import numpy as np

### specification formats for SPC header blocks.
#
# These are a collection of tuples where the first value of the tuple is
# the parameter name, and the second value is the datatype as specified
# using struct module syntax)

# header for new format spc files

# note specification for new format claims unsigned char for the exp
# field but in reality it is signed

NEWFORMAT = (
	('exper',  'B'),     # BYTE fexper;      /* Instrument technique code (see below) */
	('exp',    'b'),     # char fexp;        /* Fraction scaling exponent integer (80h=>float) */
	('npts',   'L'),     # DWORD fnpts;      /* Integer number of points (or TXYXYS directory position) */
	('first',  'd'),     # double ffirst;    /* Floating X coordinate of first point */
	('last',   'd'),     # double flast;     /* Floating X coordinate of last point */
	('nsub',   'L'),     # DWORD fnsub;      /* Integer number of subfiles (1 if not TMULTI) */
	('xtype',  'B'),     # BYTE fxtype;      /* Type of X axis units (see definitions below) */
	('ytype',  'B'),     # BYTE fytype;      /* Type of Y axis units (see definitions below) */
	('ztype',  'B'),     # BYTE fztype;      /* Type of Z axis units (see definitions below) */
	('post',   'B'),     # BYTE fpost;       /* Posting disposition (see GRAMSDDE.H) */
	('date',   'L'),     # DWORD fdate;      /* Date/Time LSB: min=6b,hour=5b,day=5b,month=4b,year=12b */
	('res',    'B'*9),   # char fres[9];     /* Resolution description text (null terminated) */
	('source', 'B'*9),   # char fsource[9];  /* Source instrument description text (null terminated) */
	('peakpt', 'H'),     # WORD fpeakpt;     /* Peak point number for interferograms (0=not known) */
	('spare',  'f'*8),   # float fspare[8];  /* Used for Array Basic storage */
	('cmnt',   'B'*130), # char fcmnt[130];  /* Null terminated comment ASCII text string */
	('catxt',  'B'*30),  # char fcatxt[30];  /* X,Y,Z axis label strings if ftflgs=TALABS */
	('logoff', 'L'),     # DWORD flogoff;    /* File offset to log block or 0 (see above) */
	('mods',   'L'),     # DWORD fmods;      /* File Modification Flags (see below: 1=A,2=B,4=C,8=D..) */
	('procs',  'B'),     # BYTE fprocs;      /* Processing code (see GRAMSDDE.H) */
	('level',  'B'),     # BYTE flevel;      /* Calibration level plus one (1 = not calibration data) */
	('sampin', 'H'),     # WORD fsampin;     /* Sub-method sample injection number (1 = first or only ) */
	('factor', 'f'),     # float ffactor;    /* Floating data multiplier concentration factor (IEEE-32) */
	('method', 'B'*48),  # char fmethod[48]; /* Method/program/data filename w/extensions comma list */
	('zinc',   'f'),     # float fzinc;      /* Z subfile increment (0 = use 1st subnext-subfirst) */
	('planes', 'L'),     # DWORD fwplanes;   /* Number of planes for 4D with W dimension (0=normal) */
	('winc',   'f'),     # float fwinc;      /* W plane increment (only if fwplanes is not 0) */
	('wtype',  'B'),     # BYTE fwtype;      /* Type of W axis units (see definitions below) */
	('reserv', 'B'*187)  # char freserv[187]; /* Reserved (must be set to zero) */
) # 510 bytes

# header for old format spc files.  Does not include the first subhdr.

OLDFORMAT = (
	('exp',    'h'),     # short oexp;
	('npts',   'f'),     # float onpts;
	('first',  'f'),     # float ofirst;
	('last',   'f'),     # float olast;
	('xtype',  'B'),     # BYTE oxtype;
	('ytype',  'B'),     # BYTE oytype;
	('year',   'H'),     # WORD oyear;
	('month',  'B'),     # BYTE omonth;
	('day',    'B'),     # BYTE oday;
	('hour',   'B'),     # BYTE ohour;
	('minute', 'B'),     # BYTE ominute;
	('res',    'B'*8),   # char ores[8];
	('peakpt', 'H'),     # WORD opeakpt;
	('nscans', 'H'),     # WORD onscans;
	('spare',  'f'*7),   # float ospare[7];
	('cmnt',   'B'*130), # char ocmnt[130];
	('catxt',  'B'*30),  # char ocatxt[30];
) # 222 bytes

# subheader format

SUBFORMAT = (
	('flgs', '  B'),  # BYTE subflgs;
	('exp',    'b'),  # char subexp;
	('index',  'H'),  # WORD subindx;
	('time',   'f'),  # float subtime;
	('next',   'f'),  # float subnext;
	('nois',   'f'),  # float subnois;
	('npts',   'L'),  # DWORD subnpts;
	('scan',   'L'),  # DWORD subscan;
	('wlevel', 'f'),  # float subwlevel;
	('resv',   'B'*4) # char subresv[4];
) # 32 bytes

# note that the LOGSTC block is not currently supported

EXP_MEANS_FLOAT = -128			# negative 128 not positive 128 (bug fixed 06/20/24)

def __bad_header(F, target):
	"""Check total number of bytes off format specification against desired value.
	
	Returns True if we do not pass the check.
	"""
	fmt = '<' + ''.join([f[-1] for f in F])
	return struct.calcsize(fmt) != target

# verify that header specifications have the right number of btyes.
# This is a check against mistakes when developing the format
# specifications.

if __bad_header(NEWFORMAT, 512 - 2):
	warnings.warn('New spchdr specification size mismatch!')

if __bad_header(OLDFORMAT, 224 - 2):
	warnings.warn('Old spchdr specification size mismatch!')
	
if __bad_header(SUBFORMAT, 32):
	warnings.warn('subhdr specification size mismatch!')

def _read_hdr(spcfile, F, endian='<'):
	"""Read header with specification format F from an open file object."""

	# initialize the output dictionary
	out = {}

	# cycle through all defined parameters in the format reading and
	# assigning the values to the output dictionary
	
	for name, fmt in F:
		n_values = len(fmt)
		
		fmt = endian + fmt

		n_bytes = struct.calcsize(fmt)
		
		out[name] = struct.unpack(fmt, spcfile.read(n_bytes))
		
		if n_values==1:
			out[name] = out[name][0]
		
	return out
	
def _read_data(spcfile, num, exp, tsprec=None, versn=None, dtype_='d', endian='<'):
	"""Read and return y-data as an ndarray from an open file object."""

	if exp==EXP_MEANS_FLOAT:
		# floating point values, everything related to fixed point
		# representation shoudl be ignored
		type_ = 'f'
		bits = None
	else:
		# fixed point representation
		if tsprec:
			bits = 16
			type_ = 'h'
		else:
			bits = 32
			type_='l'
			
	fmt = endian + type_*int(num)
	n_bytes = struct.calcsize(fmt)

	# read the data
	buf = spcfile.read(n_bytes)

	if bits==32 and versn==77:
		# reverse the words
		buf = bytes([c for t in zip(buf[2::4], buf[3::4], buf[::4], buf[1::4]) for c in t])

	data = np.array(struct.unpack(fmt, buf), dtype=dtype_)
		
	if exp != EXP_MEANS_FLOAT:
		# scaling factor to convert from fixed point back to floating point
		K = 2**(exp-bits)
		data *= K
		
	return data


class spcreader(object):
	"""Object interface to read single spcfiles.
	
	Initialize with a desired SPC filename as first input, or set the
	`spcfile` attribute directly.  The read() method will read the whole
	file and populate atributes of the object with the read data.
	
	May also iterate through spectra in the file:
	
	spectra = [s for s in spcreader(filename)]
	
	The file will automatically be opened and the header read if that
	hasn't already happened.  The file will automatically be closed upon
	a StopIteration() if the close_on_stop parameters is set (which is
	the default).
	"""
	def __init__(self, spcfile=None, dtype_='f', always_check_old_multifile=True, close_on_stop=True):
		# the spcfile may be a filename or a file-like object.  In all
		# cases, the name of the file is placed in the spcfile attribute
		# and a _file attribute is used to hold the file-like object
		self.spcfile = spcfile

		# desired output datatype of the ydata
		self._dtype = dtype_
		
		self._endian = None
		self._tflags = None
		self._versn = None
		self._TMULTI = None
		self._TXYXYS = None
				
		self._num_subfiles = None
		self._subfile_start = None
		
		self.hdr = None
		self.subhdr = None
		self.x = None
		self.y = None
		self.logstc = None		# not supported yet

		# 06/06/23 new data collected Oct 2022 uses old SPC format to
		# hold a multifile but stores the number of scans as 1.  Before,
		# I believe the check was only needed when the nscans overflowed
		# the datatype and was set at the intmax of a uint16.  Now it
		# also needs to be updated when the value of nscans equals 1.
		# So I will by default now allow always checking for the
		# inferred nscans value for an old multifile.  But it can be
		# customized.
		self.always_check_old_multifile = always_check_old_multifile

		self._close_on_stop = close_on_stop

	def __iter__(self):
		"""Make the object iterable to iterate through spectra in the file."""
		return self

	def __next__(self):
		"""Make the object iterable to iterate through spectra in the file."""
		if not self._ready:
			self.open()
			self.read_header()
		try:
			y, subhdr = self.read_block()
		except StopIteration as E:
			if self._close_on_stop:
				self.close()
			raise(E)
		except Exception as E:
			print(E)
			import pdb
			pdb.set_trace()
			raise(E)
		else:
			self.subhdr = subhdr
			return y

	@property
	def spcfile(self):
		return self._spcfile

	@spcfile.setter
	def spcfile(self, value):
		if value is None or isinstance(value, str):
			self.close()
			self._spcfile = value
			self._file = None
		elif not hasattr(value, 'closed'):
			raise(TypeError('Expected string or file-like object.'))
		else:
			self.close()
			self._file = value
			self._spcfile = value.name

	@spcfile.deleter
	def spcfile(self):
		self.close()
		self._spcfile = None
		self._file = None
			
	@property
	def closed(self):
		return (not hasattr(self, '_file')) or self._file is None or self._file.closed
		
	def open(self):
		"""If not already open, open or re-open the file named in the spcfile atribute."""
		if self.closed:
			try:
				self._file = open(self.spcfile, 'rb')
			except TypeError:
				raise(ValueError('spcfile attribute not set.'))

	def close(self):
		"""If not already closed, close the current file."""
		if not self.closed:
			self._file.close()
			self._file = None

	def __del__(self):
		self.close()

	@property	
	def _ready(self):
		return not self.closed and hasattr(self, 'hdr') and self.hdr is not None
			
	def read_header(self):
		"""Read spc header from opened file, populating the self.hdr attribute."""
		self.open()
		
		### flags and version
		self._tflags, self._versn = struct.unpack('BB', self._file.read(2))

		### read the spc header
		if self._versn != 77:
			if self._versn == 75: # new format LSB first
				self._endian = '<'
			else:
				self._endian = '>'

			hdr = _read_hdr(self._file, NEWFORMAT, self._endian)

		else:
			self._endian = '<'

			hdr = _read_hdr(self._file, OLDFORMAT, self._endian)

		self._TSPREC = self._tflags & 0x01
		self._TCGRAM = self._tflags & 0x02 # not used
		self._TMULTI = self._tflags & 0x04
		self._TRANDM = self._tflags & 0x08
		self._TORDRD = self._tflags & 0x10
		self._TALABS = self._tflags & 0x20 # ignored
		self._TXYXYS = self._tflags & 0x40
		self._TXVALS = self._tflags & 0x80

		### check for multifile
		if self._TMULTI:
			# TMULTI is set, there is more than one Y block
			if self._versn == 77:
				num_subfiles = hdr['nscans']
			else:
				num_subfiles = hdr['nsub']
		else:
			num_subfiles = 1

		### read (or calculate) x data
		if self._TXVALS:
			# X-data is unevenly spaced, and must be read
			fmt = self._endian + 'f'*hdr['npts']

			x = np.array(struct.unpack(fmt, self._file.read(struct.calcsize(fmt))))

			if self._TXYXYS:
				# different x axis for each
				self.x = []*num_subfiles
				self.x[0] = x
			else:
				self.x = x
		else:
			# simply calculate x
			self.x = np.linspace(hdr['first'], hdr['last'], int(hdr['npts']))

		if self._TXYXYS and hdr['npts'] != 0:
			warnings.warn('Directory offset with multifile not supported yet!')

		if self._versn == 77 and self._TMULTI:
			if self.always_check_old_multifile or num_subfiles==32767:
				# I have this problem with imcad data that uses the old
				# format but stores more scans than can be held in a uint16.
				# We are at the maximum number of scans, we should double
				# check the size of the file compared to the expected number
				# of scans.
				
				#remember current position
				cpos = self._file.tell()
				
				# move to end of file and get position
				self._file.seek(0, os.SEEK_END)
				eof = self._file.tell()
				
				# restore previous location
				self._file.seek(cpos, 0)

				remain = eof - cpos
				if self._TSPREC:
					B = 2
				else:
					B = 4

				num = remain/(hdr['npts']*B + 32) # 32 is size of subhdr

				if num != num_subfiles:
					print("Old spc format: 'nscans' value of %i differs from inferred value." % num_subfiles)
					
					num_subfiles = int(np.floor(num))

					left_over = num - num_subfiles
					
					print('  (Inferring the presence of %i subfiles from file size)' % num_subfiles)

					print('  (%g leftover bytes)' % left_over)
					
		self.hdr = hdr
		self._num_subfiles = num_subfiles
		self._subfile_start = self._file.tell() # added 09/05/23
		
	def read_subhdr(self):
		"""TO DO: make this a hidden method."""
		return _read_hdr(self._file, SUBFORMAT)
		
	def read_block(self):
		try:
			subhdr = self.read_subhdr()
		except Exception as E:
			# possibly at end of file
			pos = self._file.tell()
			eof = self._file.seek(0, os.SEEK_END)
			if pos==eof:
				raise(StopIteration())
			else:
				self._file.seek(pos, 0)
				raise(E)
		
		if self._num_subfiles==1:
			npts = self.hdr['npts']
			exp = self.hdr['exp']
		else:
			exp = subhdr['exp']
			if self._TXYXYS:
				npts = subhdr['npts']
			else:
				npts = self.hdr['npts']

		try:		
			y = _read_data(self._file, npts, exp, tsprec=self._TSPREC, versn=self._versn, dtype_=self._dtype, endian=self._endian)
		except Exception as E:
			# possibly at end of file
			pos = self._file.tell()
			eof = self._file.seek(0, os.SEEK_END)
			if pos==eof:
				raise(StopIteration())
			else:
				self._file.seek(pos, 0)
				raise(E)
		
		return (y, subhdr)

	def read_ydata(self):
		y, subhdr = zip(*[self.read_block() for i in range(self._num_subfiles)])
		self.y = np.stack(y)
		self.subhdr = subhdr

	def read(self):
		self.read_header()
		self.read_ydata()
		
		# logstc block?
		# reordering of multifiles?

		self.close()


def read(spcfile, **kwargs):
	"""Read an SPC file returning (hdr, subhdr, x, y)."""
	reader = spcreader(spcfile, **kwargs)
	reader.read()
	return reader.hdr, reader.subhdr, reader.x, reader.y.T
