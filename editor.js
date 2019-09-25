// cfz, September 2018.
// TODO: novelty, 200ms wait before search to reduce lag, aesthetics, cleaning up word results.

const vectorURL = "https://storage.googleapis.com/mlstorage-cloud/Data/glove.6B.50d.txt.zip"

//fetch doesn't have progress, file too big to not have indicator ==> xhr ugliness.
function fetchWordVectors(onProgress) {
  return new Promise((resolve, reject) => {
    localforage.getItem('vectorZip').then(zip => {
      if (zip == null) {
        var xhr = new XMLHttpRequest();
        xhr.open("GET", vectorURL, true);
        xhr.responseType = "arraybuffer";
        xhr.onerror = (e) => {
          console.log("✗ Error fetching vector file:\n" + xhr.statusText)
          reject(xhr.statusText)
        }
        xhr.onload = (e) => {
          let zip = xhr.response
          localforage.setItem('vectorZip', zip).then(() => {
            console.log("✓ Downloaded and stored vector file")
            resolve(zip)
          })
        }
        xhr.onprogress = (e) => {
          onProgress(e.loaded / e.total)
        }
        xhr.send();
      }
      else {
        console.log("✓ Found stored vector file")
        resolve(zip)
      }
    })
  })
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function loadWordVectors(zipArrayBuffer, onProgress) {
  let text = await JSZip.loadAsync(zipArrayBuffer).then(res => res.files["glove.6B.50d.txt"].async("string"))
    .catch(err => console.log("✗ Error unzipping vector file:\n" + err))

  let lines = text.split("\n")
  lines = lines.slice(0, -1)

  let wordDict = {fromIdx: {}, fromWord: {}}
  let wordVectors = tf.buffer([lines.length, 50], "float32")

  for (var idx in lines) {
    let line = lines[idx]
    let sep = line.indexOf(" ")
    let word = line.substring(0, sep)
    let vector = line.substring(sep+1).split(' ')
    vector = vector.map(x => parseFloat(x))

    wordDict.fromIdx[parseInt(idx)] = word
    wordDict.fromWord[word] = idx

    let magnitude = Math.sqrt(vector.reduce((acc, x) => acc + Math.pow(x, 2), 0.0))

    for (var jdx in vector) {
      let feat = vector[jdx] / magnitude
      wordVectors.set(feat, parseInt(idx), parseInt(jdx))
    }

    if (idx % Math.floor(lines.length / 100) == 0) {
      onProgress(parseInt(idx)/lines.length)
      await sleep(1) //react needs a frame to render or something like that
    }
  }

  console.log("✓ Loaded word vector matrix of shape " + wordVectors.shape)
  wordVectors = wordVectors.toTensor()
  return [wordVectors, wordDict]

}

function getHighestKIndices(array, k) {
  k = k + 1
  let highestValues = Array(k)
  for (var i=0; i<array.length; i++) {
    for (var j=0; j<highestValues.length; j++) {
      if (highestValues[j] == undefined) {
        highestValues[j] = {val: array[i], index: i}
        break
      }
      else if (array[i] > highestValues[j].val) {
        highestValues[j].val = array[i]
        highestValues[j].index = i
        break
      }
    }
  }
  highestValues.sort((a, b) => b.val - a.val)
  highestValues = highestValues.slice(1).map(x => x.index)
  return highestValues
}

class App extends React.Component {

  welcomeText = "To get started, just type!\n\nYou can bold and italicize with cmd-B and cmd-I.\n\nOnce the word vectors have initialized, word suggestions will appear as you type.\n\nProgress is saved instantly to the app - your text will be here when you come back."

  styles = {
    main: {
      height: "100%",
      width: "100%",
    },
    editor: {
      height: "100%", //no flexbox support for Draft.js editor; as a component doesn't respond well to css.
      marginTop: "-40px",
      padding: "25px",
      paddingTop: "65px",
      width: "100%",
      fontSize: 18,
      fontFamily: "Sans-Serif",
      boxSizing: "border-box",
    }
  }

  //"absolute" is positioned relative to closet parent with "relative", default position is static
  createSuggestionsDecorator(blockKey, start, end, similarWords) {
    let AddSpan = (props) => {
      return (<span id="activeWord" style={{position: "relative"}}>
        {props.children}
      </span>)
    };
    let findWords = (contentBlock, callBack, contentState) => {
      if (contentBlock.getKey() == blockKey) {
        callBack(start, end)
      }
    }
    let decorator = new Draft.CompositeDecorator([
      {
        strategy: findWords,
        component: AddSpan,
      },
    ]);
    return decorator
  }

  constructor(props) {
    super(props)
    this.blockstackSession = new blockstack.UserSession()
    if(!this.blockstackSession.isUserSignedIn() && this.blockstackSession.isSignInPending()) {
       this.blockstackSession.handlePendingSignIn().then(userData => {
         console.log("✓ Logged in as " + userData.username)
       })
     }

    localforage.setDriver(localforage.INDEXEDDB)
    tf.setBackend("cpu") //using webgl backend causes slight freeze after every big matrix op
    console.log("✓ Loaded react")
    let onProgress = loadProgress => {this.setState({loadProgress})}

    fetchWordVectors(onProgress)
      .then(buffer => {
        this.setState({loadState: ["Loading word vectors", 1], loadProgress: 0})
        return loadWordVectors(buffer, onProgress)
      }).then((wordData) => {
        this.setState({
          wordVectors: wordData[0],
          wordDict: wordData[1],
          vectorsLoaded: true,
          loadState: ["Ready", 3]
        })
      })

    let savedEditor = localStorage.getItem("savedEditor")
    if (savedEditor == null) {
      console.log("✓ Creating empty Draft editor")
      // var newEditor = Draft.EditorState.createEmpty()
      var newEditor = Draft.EditorState.createWithContent(Draft.ContentState.createFromText(this.welcomeText))
    } else {
      console.log("✓ Found saved Draft editor")
      var newEditor = Draft.EditorState.createWithContent(Draft.convertFromRaw(JSON.parse(savedEditor)))
    }

    this.state = {
      editorState: newEditor,
      activeWord: {text: "", blockKey: "", pos: null},
      similarWords: [],
      vectorsLoaded: false,
      wordDict: null,
      wordVectors: null,
      userData: null,
      loadProgress: 0.0,
      loadState: ["Downloading word vectors", 0],
      lastChange: Date.now(),
      lastChangeTimeout: null,
    }

    let save = () => {
      let rawEditorState = JSON.stringify(Draft.convertToRaw(this.state.editorState.getCurrentContent()))
      localStorage.setItem("savedEditor", rawEditorState)
    }
    setInterval(save, 2000)
  }

  handleKeyCommand(command, editorState) {
    const newState = Draft.RichUtils.handleKeyCommand(editorState, command);
    if (newState) {
      this.onChange(newState);
      return 'handled';
    }
    return 'not-handled';
  }

  onChange(editorState) {
    if (editorState.getLastChangeType() == "apply-entity") return //prevent shenanigans when editing happens in decorator

    this.setState((currentState, props) => {
      let word = this.getActiveWord(editorState)
      let wordText = word.text && word.text.replace(",", "")
      wordText = wordText && wordText.replace(".", "")
      let lastChangeType = editorState.getLastChangeType()

      // force minimum time diff for calculation when backspacing.
      let lastChangeTimeDiff = Date.now() - this.state.lastChange
      // console.log(lastChangeTimeDiff)
      let wasLongDelay = lastChangeTimeDiff > 145// || lastChangeType != 'backspace-character'
      let lastChangeTimeout
      if (!wasLongDelay) {
        let updateEditor = () => {this.onChange(this.state.editorState)}
        lastChangeTimeout = setTimeout(updateEditor, 150)
      }
      clearTimeout(this.state.lastChangeTimeout)

      let wasCharacterEdit = lastChangeType == 'insert-characters' || lastChangeType == 'backspace-character'
      let wasSelectionChange = currentState.editorState.getCurrentContent() == editorState.getCurrentContent()
      let isNewWord = wordText != currentState.activeWord.text || word.blockKey != currentState.activeWord.blockKey
        || word.wordStart !=currentState.activeWord.wordStart
      let vectorAvailible = this.state.vectorsLoaded && wordText in this.state.wordDict.fromWord

      if((wasCharacterEdit || wasSelectionChange) && isNewWord && vectorAvailible && wasLongDelay) {
        this.getWordSuggestions(wordText).then(similarWords => {
          let wordDecorator = this.createSuggestionsDecorator(word.blockKey, word.wordStart, word.wordEnd, similarWords) //doing the update async didn't help freezing up
          this.setState((currentState, props) => {
            return {
              editorState: Draft.EditorState.set(currentState.editorState, {decorator: wordDecorator}),
              similarWords,
            }
          })
        })
        return {
          editorState: Draft.EditorState.set(editorState, {decorator: null}),
          activeWord: word,
          // similarWords,
          lastChange: Date.now(),
          lastChangeTimeout
        }
      }
      else if (!isNewWord) {
        return {editorState}
      }
      else {
        return {
          editorState: Draft.EditorState.set(editorState, {decorator: null}),
          activeWord: "",
          // similarWords: [],
          lastChange: Date.now(),
          lastChangeTimeout
        }
      }
    })
  }

  getActiveWord(editorState) {
      let selectionState = editorState.getSelection()
      let blockKey = selectionState.getAnchorKey()
      let currentContent = editorState.getCurrentContent();
      let currentContentBlock = currentContent.getBlockForKey(blockKey);
      let text = currentContentBlock.getText()
      let start = selectionState.getStartOffset()
      let end = selectionState.getEndOffset()
      let wordStart = text.lastIndexOf(" ", start-2) + 1
      let wordEnd = text.indexOf(" ", start-1)
      let word
      if (start != end) {
        word = null
      }
      else {
        if (wordStart == -1) wordStart = 0
        if (wordEnd == -1) wordEnd = text.length
        word = text.substring(wordStart, wordEnd).toLowerCase()
      }
      return {text: word, blockKey, wordStart, wordEnd}
  }

  async getWordSuggestions(word) {
    //make word vector
    let wordIndex = this.state.wordDict.fromWord[word]
    let wordVectorsBuffer = await this.state.wordVectors.buffer()
    let vectorLength = this.state.wordVectors.shape[1]
    let vector = tf.buffer([vectorLength, 1], "float32")
    for (var idx = 0; idx < this.state.wordVectors.shape[1]; idx++) {
      vector.set(wordVectorsBuffer.get(wordIndex, idx), idx, 0)
    }
    vector = vector.toTensor()
    //get similarities
    let similarities = tf.matMul(this.state.wordVectors, vector)
    let similarityData = await similarities.data()
    similarities.dispose()
    vector.dispose()
    let similarWordIndexes = getHighestKIndices(similarityData, 10)
    let similarWords = similarWordIndexes.map(x => this.state.wordDict.fromIdx[x])
    return similarWords
  }

  componentDidMount() {
  }

  componentDidUpdate() {
    let currentWord = document.getElementById("activeWord")
    let lastPos = this.state.activeWord.pos
    if (currentWord) {
      let pos = currentWord.getBoundingClientRect()
      pos.y += window.scrollY
      if (lastPos == null || (pos.x != lastPos.x && pos.y != lastPos.y)) {
        this.setState({activeWord: {...this.state.activeWord, pos}})
      }
    }
    else if (!currentWord && lastPos != null) {
        this.setState({activeWord: {...this.state.activeWord, pos: null}})
    }
  }

  render() {
    let doRenderDropdown = this.state.activeWord.pos != null
    if (doRenderDropdown) {
      var x = this.state.activeWord.pos.x
      var  y = this.state.activeWord.pos.y + 23
    }
    var mainAppArea = (
      <div style={this.styles.main}>
        <StatusBar state={this.state.loadState} progress={this.state.loadProgress} loaded={this.state.vectorsLoaded} logout={() => this.blockstackSession.signUserOut()} />
        <div style={this.styles.editor}>
          <Draft.Editor
            editorState={this.state.editorState}
            onChange={this.onChange.bind(this)}
            handleKeyCommand={this.handleKeyCommand.bind(this)}
            style={{}}
          />
        </div>
        {doRenderDropdown && <Dropdown similarWords={this.state.similarWords} position={{x, y}}/>}
      </div>
    )

    return (
      this.blockstackSession.isUserSignedIn() ? mainAppArea : <LandingPage/>
    )
  }
}

class Dropdown extends React.Component {

  style = {
    dropdown: {
      fontFamily: "Sans-Serif",
      borderRadius: "7px",
      boxShadow: "0px 1px 4px #ccc",
      overflow: "hidden",
      position: "absolute",
      backgroundColor: "white",
      zIndex: 100,
      MozUserSelect:"none",
      WebkitUserSelect:"none",
    },
    list: {
      listStyle: "none",
      padding: 0,
      margin: 0,
      fontSize: 15,
      boxSizing: "border-box",
    },
    item: {
      marginBottom: "-1px",
      borderBottom: "1px solid #ccc",
      padding: "2px",
      paddingLeft: "4px",
      paddingRight: "4px",
    }
  }

  constructor(props) {
    super(props)
  }

  render() {
    return (
      <div style={{top: this.props.position.y, left: this.props.position.x, ...this.style.dropdown}}>
        <ul style={this.style.list}>
        {this.props.similarWords.map((word) =>
          <div key={word}>
            <li style={this.style.item}>{word}</li>
          </div>
        )}
        </ul>
      </div>
    )
  }
}

class StatusBar extends React.Component {

  style = {
    bar: {
      backgroundColor: "#222",
      fontFamily: "Sans-Serif",
      height: "40px",
      width: "100%",
      color: "white",
      display: "flex",
      flexDirection: "row",
      justifyContent: "space-between",
      alignItems: "center",
      paddingLeft: "25px",
      paddingRight: "25px",
      boxSizing: "border-box",
      fontSize: "16px",
      boxShadow: "0px 1px 3px #ccc",
    }
  }

  buttonStyle = {
    color: "white",
    backgroundColor: "rgba(0, 0, 0, 0)",
    borderRadius: "2px 2px 2px 2px",
    fontSize: "16px",
    borderColor: "#000",
    padding: "4 6",
    cursor: "pointer",
    borderWidth: "1px 1px 1px 1px",
    borderStyle: "solid",
    borderColor: "white"
    // width: "67px",
  }

  constructor(props) {
    super(props)
  }
  render () {
    let doShowProgress = this.props.state[1] == 0 || this.props.state[1] == 1
    let progress = "(" + Math.round(this.props.progress * 100).toString() + "%)"
    return (
      <div style={this.style.bar}>
        <div>{this.props.state[0] + " "} {doShowProgress && progress}</div>
        <div><button style={this.buttonStyle} onClick={() => this.props.logout()}>Logout</button></div>
      </div>
    )
  }
}

class LandingPage extends React.Component {

  appConfig = new blockstack.AppConfig(['store_write'])

  exampleStyle = {
    sentence: {
      fontSize: "24px",
      fontFamily: "sans-serif",
    },
    dropdown: {
      fontFamily: "Sans-Serif",
      borderRadius: "7px",
      boxShadow: "0px 1px 4px #ccc",
      // overflow: "hidden",
      backgroundColor: "white",
      zIndex: 100,
      marginTop: "1.9em",
      position: "relative",
      left: "-3.8em",
      marginRight: "-3.8em",
      // MozUserSelect:"none",
      // WebkitUserSelect:"none",
    },
    list: {
      listStyle: "none",
      padding: 0,
      margin: 0,
      fontSize: 20,
      boxSizing: "border-box",
    },
    item: {
      marginBottom: "-1px",
      borderBottom: "1px solid #ccc",
      padding: "4px",
      paddingLeft: "6px",
      paddingRight: "6px",
    }
  }

  headerStyle = {
    fontFamily: "LyonDisplay,Georgia,serif",
    fontSize: "60px",
    marginBottom: "0px",
    marginTop: "20px",
    // letterSpacing: "0.01em",
  }

  buttonStyle = {
    fontFamily: "LyonDisplay,Georgia,serif",
    color: "black",
    fontSize: "24px",
    backgroundColor: "white",
    borderRadius: "2px 2px 2px 2px",
    borderColor: "#000",
    padding: "15 30",
    cursor: "pointer",
    borderWidth: "1px 1px 1px 1px",
    borderStyle: "solid",
    // width: "67px",
  }

  constructor(props) {
    super(props)
    this.userSession = new blockstack.UserSession({appConfig: this.appConfig})
    this.state = {
      blink: true,
    }
    setTimeout(() => this.toggleBlink(), 500)
  }

  signin() {
    this.userSession.redirectToSignIn()
  }

  toggleBlink() {
    this.setState(oldState => {return {blink: !oldState.blink}})
    setTimeout(() => this.toggleBlink(), 500)
  }

  render() {
    return (
      <div style={{display: "flex", flexDirection: "column", alignItems:"center", boxSizing: "border-box", margin: "20px"}}>
        <h1 style={this.headerStyle}>Prosepaper</h1>

        <div style={{display: "flex", flexDirection: "row", marginTop: "130px", marginBottom: "120px"}}>

          <div style={this.exampleStyle.sentence}>A word processor that actually knows a bit about words <span style={{color: this.state.blink ? "black" : "white"}}>|</span>
          <br></br><br></br><br></br><br></br><br></br><br></br>


          </div>

          <div style={this.exampleStyle.dropdown}>
            <ul style={this.exampleStyle.list}>
              <div><li style={this.exampleStyle.item}>phrases</li></div>
              <div><li style={this.exampleStyle.item}>uttered</li></div>
              <div><li style={this.exampleStyle.item}>language</li></div>
              <div><li style={this.exampleStyle.item}>expression</li></div>
              <div><li style={this.exampleStyle.item}>letters</li></div>
              <div><li style={this.exampleStyle.item}>write</li></div>
              <div><li style={this.exampleStyle.item}>text</li></div>
              <div><li style={this.exampleStyle.item}>references</li></div>
              <div><li style={this.exampleStyle.item}>describes</li></div>
              <div><li style={this.exampleStyle.item}>meaning</li></div>
            </ul>
          </div>

        </div>

        <div style={{...this.exampleStyle.sentence, fontFamily: "LyonDisplay,Georgia,serif", marginBottom: "60px"}}><b>Get realtime word suggestions while typing</b></div>
        <button style={this.buttonStyle} onClick={() => this.signin()}>Sign in with Blockstack
        </button>
      </div>
    )
  }
}

ReactDOM.render(<App />, document.getElementById("root"))
