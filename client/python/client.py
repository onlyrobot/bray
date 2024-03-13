import requests, uuid

class Client:
    def __init__(self, host, port, timeout=60):
        self.timeout = timeout
        self.session = ""
        self.sess = requests.Session()
        self.url = "http://{}:{}/step".format(host, port)

    def _request(self, kind, data):
        res = self.sess.post(
            url=self.url,
            headers={
                "session": self.session,
                "step_kind": kind,
            },
            data=data,
            timeout=self.timeout,
        )
        if res.status_code != 200:
            raise Exception(res.text)
        return res.content

    def start(self, game="", agent=""):
        if game == "":
            game = str(uuid.uuid4())
        self.session = game
        if agent:
            self.session = game + "-agent-" + agent
        self._request("start", b"")

    def tick(self, data):
        return self._request("tick", data)

    def stop(self):
        return self._request("stop", "")
    

if __name__ == "__main__":
    client = Client("apps-hp.danlu.netease.com", 36390)
    client.start()
    for i in range(10):
        print(client.tick("hello world"))
    client.stop()