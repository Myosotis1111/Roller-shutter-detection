class Window:
    def __init__(self, doc, status, doc_max=None, doc_min=None, count_open=0, count_closed=0):
        self._doc = doc  # The degree of Closure of the roller shutter
        self._status = status  # The status of the roller shutter, can be open, closed, blocked or not_detected
        self._doc_max = doc_max  # The maximum possible DoC
        self._doc_min = doc_min  # The minimum possible DoC
        self.button = None  # Corresponding button No. in GUI.
        self._count_open = count_open  # The number of open status detected through the detection period
        self._count_closed = count_closed  # The number of close status detected through the detection period

    @property
    def doc(self):
        return self._doc

    @property
    def doc_max(self):
        return self._doc_max

    @property
    def doc_min(self):
        return self._doc_min

    @property
    def status(self):
        return self._status

    @property
    def count_open(self):
        return self._count_open

    @property
    def count_closed(self):
        return self._count_closed

    @status.setter
    def status(self, value):
        self._status = value

    @doc_max.setter
    def doc_max(self, value):
        self._doc_max = value

    @doc_min.setter
    def doc_min(self, value):
        self._doc_min = value

    @doc.setter
    def doc(self, value):
        self._doc = value

    @count_open.setter
    def count_open(self, value):
        self._count_open = value

    @count_closed.setter
    def count_closed(self, value):
        self._count_closed = value
